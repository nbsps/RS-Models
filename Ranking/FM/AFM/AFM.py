import os
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

data_path = '../../../dataset/criteo/train.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
EMBEDDING_DIM = 48
LEARNING_RATE = 5e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 1024
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


class Dnn(nn.Module):
    def __init__(self, hidden_units, dropout=0.):
        super(Dnn, self).__init__()
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(dropout)

        for i in self.dnn_network:
            i.weight.data.normal_(0, 0.05)

    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = torch.relu(x)
        x = self.dropout(x)
        return x


class Attention_layer(nn.Module):
    def __init__(self, att_units):
        super(Attention_layer, self).__init__()
        self.att_w = nn.Linear(att_units[0], att_units[1])
        self.att_dense = nn.Linear(att_units[1], 1)

        self.att_w.weight.data.normal_(0, 0.05)
        self.att_dense.weight.data.normal_(0, 0.05)

    def forward(self, bi_interaction):
        a = self.att_w(bi_interaction)
        a = torch.relu(a)
        att_scores = self.att_dense(a)
        att_weight = torch.softmax(att_scores, dim=1)
        att_out = torch.sum(att_weight * bi_interaction, dim=1)
        return att_out


class AFM(nn.Module):
    def __init__(self, feature_columns, hidden_units, emb_dim=8, att_vector=8, dropout=0.5):
        super(AFM, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols, self.sparse_feas_map = feature_columns
        self.fea_num = len(self.dense_feature_cols) + emb_dim
        self.num_fields = len(feature_columns)
        hidden_units.insert(0, self.fea_num)
        # sparse embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=emb_dim)
            for key, val in self.sparse_feas_map.items()
        })
        self.attention = Attention_layer([emb_dim, att_vector])
        self.bn = nn.BatchNorm1d(self.fea_num)
        self.dnn_network = Dnn(hidden_units, dropout)
        self.nn_final_linear = nn.Linear(hidden_units[-1], 1)

        for k, i in self.embed_layers.items():
            i.weight.data.normal_(0, 0.05)
        self.nn_final_linear.weight.data.normal_(0, 0.05)

    def forward(self, sx, dx):
        dense_inputs, sparse_inputs = dx, sx
        sparse_embeds = [self.embed_layers['embed_' + key](sparse_inputs[:, i])
                         for key, i in zip(self.sparse_feas_map.keys(), range(sparse_inputs.shape[1]))]
        sparse_embeds = torch.stack(sparse_embeds)
        sparse_embeds = sparse_embeds.permute((1, 0, 2))
        i1, i2 = [], []
        for i in range(self.num_fields):
            for j in range(i + 1, self.num_fields):
                i1.append(i)
                i2.append(j)
        p = sparse_embeds[:, i1, :]
        q = sparse_embeds[:, i2, :]
        bi_interaction = p * q
        att_out = self.attention(bi_interaction)
        x = torch.cat([att_out, dense_inputs], dim=-1)
        x = self.bn(x)
        dnn_outputs = self.nn_final_linear(self.dnn_network(x))
        outputs = torch.sigmoid(dnn_outputs)
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
    mms = MinMaxScaler()
    data_df[dense_features] = mms.fit_transform(data_df[dense_features])
    # discrete -> continuous numbers (continuous feature)
    data_df[sparse_features] = OrdinalEncoder().fit_transform(data_df[sparse_features])
    y = np.array(data_df['Label'])

    train_X, test_X, train_y, test_y = train_test_split(data_df, y, train_size=0.7, random_state=2022)
    train_dataset = CriteoDataset(np.array(train_X[sparse_features]), np.array(train_X[dense_features]), train_y)
    test_dataset = CriteoDataset(np.array(test_X[sparse_features]), np.array(test_X[dense_features]), test_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    hidden_units = [128, 64, 32]
    sparse_feas_map = {}
    for key in sparse_features:
        sparse_feas_map[key] = data_df[key].nunique()
    feature_info = [dense_features, sparse_features, sparse_feas_map]
    model = AFM(feature_info, hidden_units, emb_dim=EMBEDDING_DIM, dropout=0.).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    loss_func = nn.BCELoss().to(device)

    train_loss_list = []
    test_auc_list = []
    for epoch in range(EPOCH):
        model.train()
        total_loss, total_len = 0, 0
        for sx, dx, y in train_loader:
            optimizer.zero_grad()
            sx, dx, y = sx.type(torch.int).to(device), dx.type(torch.float32).to(device), y.type(torch.float32).to(
                device)
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
                sx, dx, y = sx.type(torch.int).to(device), dx.type(torch.float32).to(device), y.type(torch.float32).to(
                    device)
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
