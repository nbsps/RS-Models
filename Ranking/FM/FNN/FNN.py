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


data_path = '../../../dataset/criteo/train.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
EMBEDDING_DIM = 64
LEARNING_RATE = 5e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 1024
EPOCH = 200


# declare Criteo Dataset
class CriteoDataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def __len__(self):
        return len(self.feature)


class MultiLayerPerceptron(nn.Module):
    def __init__(self, layer, batch_norm=True):
        super(MultiLayerPerceptron, self).__init__()
        layers = []
        input_size = layer[0]
        for output_size in layer[1: -1]:
            layers.append(nn.Linear(input_size, output_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(output_size))
            layers.append(nn.ReLU())
            input_size = output_size
        layers.append(nn.Linear(input_size, layer[-1]))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# define FM Model
class FactorizationMachine(nn.Module):
    def __init__(self, field_dims, embed_dim=4):
        super(FactorizationMachine, self).__init__()

        self.embed1 = Embedding(sum(field_dims), 1)
        self.embed2 = Embedding(sum(field_dims), embed_dim)
        self.bias = nn.Parameter(torch.zeros((1,)), requires_grad=True)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1],), dtype=np.int)

        self.embed1.weight.data.uniform_(0, 0.005)
        self.embed2.weight.data.uniform_(0, 0.005)

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        # embed(x) shape: (batch_size, num_fields, embed_dim)
        x = x + x.new_tensor(self.offsets)
        square_sum = self.embed2(x).sum(dim=1).pow(2).sum(dim=1)
        sum_square = self.embed2(x).pow(2).sum(dim=1).sum(dim=1)
        output = self.embed1(x).squeeze(-1).sum(dim=1) + self.bias + (square_sum - sum_square) / 2
        output = torch.sigmoid(output)
        return output


# define FNN Model
class FactorizationMachineSupportedNeuralNetwork(nn.Module):
    def __init__(self, field_dims, embed_dim=4):
        super(FactorizationMachineSupportedNeuralNetwork, self).__init__()
        self.embed1 = Embedding(sum(field_dims), 1)
        self.embed2 = Embedding(sum(field_dims), embed_dim)
        self.mlp = MultiLayerPerceptron([(embed_dim + 1) * len(field_dims), 128, 64, 32, 1])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1],), dtype=np.int)

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        x = x + x.new_tensor(self.offsets)
        w = self.embed1(x).squeeze(-1)
        v = self.embed2(x).reshape(x.shape[0], -1)
        stacked = torch.hstack([w, v])
        output = self.mlp(stacked)
        output = torch.sigmoid(output)
        return output.squeeze()


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
    train_loader = DataLoader(train_dataset, batch_size=1024)
    test_loader = DataLoader(test_dataset, batch_size=1024)

    model = FactorizationMachine(field_dims, EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    loss_func = nn.BCELoss().to(device)

    """ FM training """
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

    """ FNN """
    fnn = FactorizationMachineSupportedNeuralNetwork(field_dims, EMBEDDING_DIM).to(device)

    # ????????????
    fm_state_dict = model.state_dict()
    fnn_state_dict = fnn.state_dict()
    fnn_state_dict['embed1.weight'] = fm_state_dict['embed1.weight']
    fnn_state_dict['embed2.weight'] = fm_state_dict['embed2.weight']
    fnn_state_dict['mlp.mlp.0.bias'] = torch.zeros_like(fnn_state_dict['mlp.mlp.0.bias']).fill_(
        fm_state_dict['bias'].item())
    fnn.load_state_dict(fnn_state_dict)

    optimizer = optim.Adam(fnn.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    criterion = nn.BCELoss()

    train_loss_list = []
    test_auc_list = []
    for epoch in range(EPOCH):
        model.train()
        total_loss, total_len = 0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            x, y = x.type(torch.int).to(device), y.type(torch.float32).to(device)
            predict = fnn(x)
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
                predict = fnn(x)
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
