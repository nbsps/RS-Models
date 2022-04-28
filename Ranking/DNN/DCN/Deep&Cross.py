import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, MinMaxScaler
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

data_path = '../../../dataset/criteo/train.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
EMBEDDING_DIM = 24
LEARNING_RATE = 5e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 256
EPOCH = 100


# declare Criteo Dataset
class CriteoDataset(Dataset):
    def __init__(self, sparse_feature, dense_feature, label):
        self.sparse_feature = sparse_feature
        self.dense_feature = dense_feature
        self.label = label

    def __getitem__(self, index):
        return self.sparse_feature[index], self.dense_feature[index], self.label[index]

    def __len__(self):
        return len(self.label)


class CrossNetwork(nn.Module):
    def __init__(self, layer_num, input_dim):
        super(CrossNetwork, self).__init__()
        self.layer_num = layer_num
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.rand(input_dim, 1))
            for i in range(self.layer_num)
        ])
        self.cross_bias = nn.ParameterList([
            nn.Parameter(torch.rand(input_dim, 1))
            for i in range(self.layer_num)
        ])

        for i in self.cross_weights:
            i.data.normal_(0, 0.005)
        for i in self.cross_bias:
            i.data.normal_(0, 0.005)

    def forward(self, x):
        x_0 = torch.unsqueeze(x, dim=2)
        x = x_0.clone()
        xT = x_0.clone().permute((0, 2, 1))
        for i in range(self.layer_num):
            x = torch.matmul(torch.bmm(x_0, xT), self.cross_weights[i]) + self.cross_bias[i] + x
            xT = x.clone().permute((0, 2, 1))
        return x.squeeze()


class Dnn(nn.Module):
    def __init__(self, hidden_units, dropout=0.):
        super(Dnn, self).__init__()
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(p=dropout)

        for i in self.dnn_network:
            i.weight.data.normal_(0, 0.005)

    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        return self.dropout(x)


class DCN(nn.Module):
    def __init__(self, feature_columns, hidden_units, layer_num, emb_dim=10):
        super(DCN, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols, self.sparse_feas_map = feature_columns
        # embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=emb_dim)
            for key, val in self.sparse_feas_map.items()
        })
        hidden_units.insert(0, len(self.dense_feature_cols) + len(self.sparse_feature_cols) * emb_dim)
        self.dnn_network = Dnn(hidden_units)
        self.cross_network = CrossNetwork(layer_num, hidden_units[0])
        self.final_linear = nn.Linear(hidden_units[-1] + hidden_units[0], 1)

        for k, i in self.embed_layers.items():
            i.weight.data.normal_(0, 0.005)
        self.final_linear.weight.data.normal_(0, 0.005)

    def forward(self, sx, dx):
        dense_input, sparse_inputs = dx, sx
        sparse_embeds = [self.embed_layers['embed_' + key](sparse_inputs[:, i]) for key, i in
                         zip(self.sparse_feas_map.keys(), range(sparse_inputs.shape[1]))]
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)
        x = torch.cat([sparse_embeds, dense_input], axis=-1)
        # Cross
        cross_out = self.cross_network(x)
        # Deep
        deep_out = self.dnn_network(x)
        final_x = torch.cat([cross_out, deep_out], axis=-1)
        final_x = self.final_linear(final_x)
        outputs = torch.sigmoid(final_x)

        return outputs.squeeze()


if __name__ == "__main__":
    data_df = pd.read_csv(data_path)
    # remove Id column
    data_df = data_df.iloc[:, 1:]
    # discrete & continuous
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    features = sparse_features + dense_features

    # fill [na]
    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    data_df[dense_features] = data_df[dense_features].fillna(0)

    # continuous feature: 归一化
    # mms = MinMaxScaler()
    # data_df[dense_features] = mms.fit_transform(data_df[dense_features])
    # continuous feature: bin discretize (large auc)
    est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    data_df[dense_features] = est.fit_transform(data_df[dense_features])
    # discrete -> continuous numbers (continuous feature)
    data_df[sparse_features] = OrdinalEncoder().fit_transform(data_df[sparse_features])
    y = np.array(data_df['Label'])

    train_X, test_X, train_y, test_y = train_test_split(data_df, y, train_size=0.7, random_state=2022)
    train_dataset = CriteoDataset(np.array(train_X[sparse_features]), np.array(train_X[dense_features]), train_y)
    test_dataset = CriteoDataset(np.array(test_X[sparse_features]), np.array(test_X[dense_features]), test_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    hidden_units = [256, 128, 64, 32]
    sparse_feas_map = {}
    for key in sparse_features:
        sparse_feas_map[key] = data_df[key].nunique()
    feature_info = [dense_features, sparse_features, sparse_feas_map]
    model = DCN(feature_info, hidden_units, 5, emb_dim=EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    loss_func = nn.BCELoss().to(device)

    train_loss_list = []
    test_auc_list = []
    for epoch in range(EPOCH):
        model.train()
        total_loss, total_len = 0, 0
        for sx, dx, y in train_loader:
            optimizer.zero_grad()
            sx, dx, y = sx.type(torch.int).to(device), dx.type(torch.float32).to(device), y.type(torch.float32).to(device)
            predict = model(sx, dx)
            loss = loss_func(predict, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total_len += len(y)
        train_loss = total_loss / total_len
        train_loss_list.append(train_loss)

        model.eval()
        labels, predicts = [], []
        with torch.no_grad():
            for sx, dx, y in test_loader:
                sx, dx, y = sx.type(torch.int).to(device), dx.type(torch.float32).to(device), y.type(torch.float32).to(device)
                predict = model(sx, dx)
                labels.extend(y.tolist())
                predicts.extend(predict.tolist())
        auc = metrics.roc_auc_score(np.array(labels), np.array(predicts))
        test_auc_list.append(auc)
        print("epoch {}, train loss is {:.4f}, test auc is {:.4f}".format(epoch, train_loss, auc))

    print("max auc in test dataset: {:.4f}".format(max(test_auc_list)))
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(test_auc_list, label='test_auc')
    plt.legend()
    plt.show()

