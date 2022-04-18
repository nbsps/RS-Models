import os
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from torch import nn, optim
from torch.nn import Embedding
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

data_path = '../../dataset/criteo/train.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
EMBEDDING_DIM = 8
LEARNING_RATE = 5e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 100
TRIAL = 100


# declare Criteo Dataset
class CriteoDataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def __len__(self):
        return len(self.feature)


# declare FM Model
class FieldAwareFactorizationMachine(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(FieldAwareFactorizationMachine, self).__init__()

        self.field_dims = field_dims

        self.bias = nn.Parameter(torch.zeros((1,)), requires_grad=True)
        self.embed_linear = nn.Embedding(sum(field_dims), 1)
        self.embed_cross = nn.ModuleList([nn.Embedding(sum(field_dims), embed_dim) for _ in field_dims])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1],), dtype=np.int)

        self.embed_linear.weight.data.uniform_(0, 0.005)
        for i in self.embed_cross:
            i.weight.data.uniform_(0, 0.005)

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        # embed(x) shape: (batch_size, num_fields, embed_dim)

        x = x + x.new_tensor(self.offsets)
        num_fields = len(self.field_dims)
        embeddings = [embed(x) for embed in self.embed_cross]
        # embeddings shape: (batch_size, num_fields(域) * num_fields(特征), embed_dim)
        embeddings = torch.hstack(embeddings)

        i1, i2 = [], []
        for i in range(num_fields):
            for j in range(i + 1, num_fields):
                i1.append(j * num_fields + i)
                i2.append(i * num_fields + j)

        # feature 0 在域 1 隐向量 x feature 1 在域 0 隐向量
        embedding_cross = torch.mul(embeddings[:, i1], embeddings[:, i2]).sum(dim=2).sum(dim=1, keepdim=True)
        output = self.embed_linear(x).sum(dim=1) + self.bias + embedding_cross
        output = torch.sigmoid(output)
        return output.squeeze(1)


if __name__ == "__main__":
    # dataset = CriteoDataset(, sample_num=100000)
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
    train_loader = DataLoader(train_dataset, batch_size=1024)
    test_loader = DataLoader(test_dataset, batch_size=1024)

    model = FieldAwareFactorizationMachine(field_dims, EMBEDDING_DIM).to(device)
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
        print("epoch {}, train loss is {}, test auc is {}".format(epoch, train_loss, auc))

    print("max auc in test dataset: ".format(max(test_auc_list)))
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(test_auc_list, label='test_auc')
    plt.legend()
    plt.show()
