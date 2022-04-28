import os
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

data_path = '../../../dataset/criteo/train.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
EMBEDDING_DIM = 64
LEARNING_RATE = 5e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 1024
EPOCH = 100


# declare Criteo Dataset
class CriteoDataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def __len__(self):
        return len(self.feature)


class DNN(nn.Module):
    def __init__(self, hidden_units, dropout=0.):
        super(DNN, self).__init__()
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])

        for i in self.dnn_network:
            i.weight.data.normal_(0, 0.005)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        return self.dropout(x)


class ProductLayer(nn.Module):
    def __init__(self, embed_dim, field_num, hidden_units):
        super(ProductLayer, self).__init__()
        self.w_z = nn.Parameter(torch.rand([field_num, embed_dim, hidden_units[0]]))
        self.w_p = nn.Parameter(torch.rand([embed_dim, embed_dim, hidden_units[0]]))
        self.l_b = torch.rand([hidden_units[0], ], requires_grad=True).to(device)
        self.w_z.data.normal_(0, 0.005)
        self.w_p.data.normal_(0, 0.005)
        self.l_b.data.normal_(0, 0.005)

    def forward(self, z, sparse_embeds):
        l_z = torch.mm(z.reshape(z.shape[0], -1),
                       self.w_z.permute((2, 0, 1)).reshape(self.w_z.shape[2], -1).T)
        f_sum = torch.unsqueeze(torch.sum(sparse_embeds, dim=1), dim=1)
        p = torch.matmul(f_sum.permute((0, 2, 1)), f_sum)
        l_p = torch.mm(p.reshape(p.shape[0], -1),
                       self.w_p.permute((2, 0, 1)).reshape(self.w_p.shape[2], -1).T)
        output = l_p + l_z + self.l_b
        return output


class PNN(nn.Module):
    def __init__(self, feature_info, hidden_units, dnn_dropout=0., embed_dim=10, outdim=1):
        super(PNN, self).__init__()
        self.dense_feas, self.sparse_feas, self.sparse_feas_map = feature_info
        self.field_num = len(self.sparse_feas)
        self.dense_num = len(self.dense_feas)
        self.embed_dim = embed_dim
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=self.embed_dim)
            for key, val in self.sparse_feas_map.items()
        })
        self.product = ProductLayer(embed_dim, self.field_num, hidden_units)
        hidden_units[0] += self.dense_num
        self.dnn_network = DNN(hidden_units, dnn_dropout)
        self.dense_final = nn.Linear(hidden_units[-1], 1)
        self.dense_final.weight.data.normal_(0, 0.005)
        for k, i in self.embed_layers.items():
            i.weight.data.normal_(0, 0.005)

    def forward(self, x):
        dense_inputs, sparse_inputs = x[:, 26:], x[:, :26]
        sparse_embeds = [self.embed_layers['embed_' + key](sparse_inputs[:, i]) for key, i in
                         zip(self.sparse_feas_map.keys(), range(sparse_inputs.shape[1]))]
        sparse_embeds = torch.stack(sparse_embeds, dim=1)
        sparse_inputs = self.product(sparse_embeds, sparse_embeds)
        l1 = torch.relu(torch.cat([sparse_inputs, dense_inputs], axis=-1))
        dnn_x = self.dnn_network(l1)
        outputs = torch.sigmoid(self.dense_final(dnn_x))
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

    # continuous feature: bin discretize
    est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    data_df[dense_features] = est.fit_transform(data_df[dense_features])

    # discrete -> continuous numbers (continuous feature)
    data_df[features] = OrdinalEncoder().fit_transform(data_df[features])

    data = np.array(data_df[features])
    y = np.array(data_df['Label'])
    field_dims = (data.max(axis=0).astype(int) + 1).tolist()

    train_X, test_X, train_y, test_y = train_test_split(data, y, train_size=0.7, random_state=2022)
    train_dataset = CriteoDataset(train_X, train_y)
    test_dataset = CriteoDataset(test_X, test_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    hidden_units = [256, 128, 64, 32]
    sparse_feas_map = {}
    for key in sparse_features:
        sparse_feas_map[key] = data_df[key].nunique()
    feature_info = [dense_features, sparse_features, sparse_feas_map]
    model = PNN(feature_info, hidden_units, embed_dim=EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    loss_func = nn.BCELoss().to(device)

    train_loss_list = []
    test_auc_list = []
    for epoch in range(EPOCH):
        model.train()
        total_loss, total_len = 0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            x, y = x.type(torch.int).to(device), y.type(torch.float32).to(device)
            predict = model(x)
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
            for x, y in test_loader:
                x, y = x.type(torch.int).to(device), y.type(torch.float32).to(device)
                predict = model(x)
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
