import os
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from torch import nn, optim
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


# declare residual block
class ResidualUnit(nn.Module):
    def __init__(self, inner_size, input_size):
        super(ResidualUnit, self).__init__()
        self.fc1 = nn.Linear(input_size, inner_size)
        self.fc2 = nn.Linear(inner_size, input_size)

        self.fc1.weight.data.normal_(0, 0.005)
        self.fc2.weight.data.normal_(0, 0.005)

    def forward(self, x):
        output = self.fc1(x)
        output = torch.relu(output)
        output = self.fc2(output)
        output = output + x
        return output


# declare DeepCrossing Network
class DeepCrossing(nn.Module):
    def __init__(self, feature_info, hidden_units, dropout=0., embed_dim=10, output_dim=1):
        super(DeepCrossing, self).__init__()
        self.dense_feas, self.sparse_feas, self.sparse_feas_map = feature_info
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=embed_dim)
            for key, val in self.sparse_feas_map.items()
        })
        embed_dim_sum = embed_dim * len(self.sparse_feas)
        dim_stack = len(self.dense_feas) + embed_dim_sum
        self.res_layers = nn.ModuleList([
            ResidualUnit(unit, dim_stack) for unit in hidden_units
        ])
        self.res_dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim_stack, output_dim)

        for k, i in self.embed_layers.items():
            i.weight.data.normal_(0, 0.005)
        self.linear.weight.data.normal_(0, 0.005)

    def forward(self, x):
        dense_inputs, sparse_inputs = x[:, 26:], x[:, :26]
        sparse_inputs = sparse_inputs.long()
        sparse_embeds = [self.embed_layers['embed_' + key](sparse_inputs[:, i]) for key, i in
                         zip(self.sparse_feas_map.keys(), range(sparse_inputs.shape[1]))]
        sparse_embed = torch.cat(sparse_embeds, axis=-1)
        r = torch.cat([sparse_embed, dense_inputs], axis=-1)
        for res in self.res_layers:
            r = res(r)
        r = self.res_dropout(r)
        outputs = torch.sigmoid(self.linear(r))
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
    model = DeepCrossing(feature_info, hidden_units).to(device)
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
