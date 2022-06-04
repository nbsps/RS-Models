import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset, DataLoader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
data_path = "../../../dataset/census-income/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pd.options.mode.chained_assignment = None
HIDDEN_UNITS = [128, 64, 32, 1]
EXPERT_UNITS = [128, 64, 32]
REGULARIZATION = 1e-6
LEARNING_RATE = 5e-4
EMBEDDING_DIM = 64
MMOE_DIM = 128
BATCH_SIZE = 64
EPOCH = 100


class CensusDataset(Dataset):
    def __init__(self, data):
        categorical_columns = ['workclass', 'education', 'occupation', 'relationship', 'race', 'sex', 'native_country']
        continuous_columns = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        self.categorical_features = data[categorical_columns].values
        self.continuous_features = data[continuous_columns].values
        self.label1 = data['income_50k'].values
        self.label2 = data['marital_status'].values

    def __getitem__(self, index):
        label1 = self.label1[index]
        label2 = self.label2[index]
        return self.categorical_features[index], self.continuous_features[index], label1, label2

    def __len__(self):
        return len(self.label1)


class Expert(nn.Module):
    def __init__(self, layers):
        super(Expert, self).__init__()
        self.mlp = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])
        for l in self.mlp:
            l.weight.data.uniform_(0, 0.005)

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x).to(device)
        return x


class DNNTower(nn.Module):
    def __init__(self, layers):
        super(DNNTower, self).__init__()
        self.mlp = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])
        for l in self.mlp:
            l.weight.data.uniform_(0, 0.005)

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x).to(device)
        return x


class MMOE(nn.Module):
    def __init__(self, feature_nums, continuous_nums, emb_dim, hidden_units, experts_units,
                 mmoe_hidden_dim=MMOE_DIM, n_expert=3, task_num=2):
        super(MMOE, self).__init__()
        self.task_num = task_num
        self.embeddings = nn.Embedding(sum(feature_nums), emb_dim)
        embedding_size = emb_dim * len(feature_nums) + continuous_nums
        self.offsets = np.array((0, *np.cumsum(feature_nums)[:-1],), dtype=np.int)
        # experts
        self.experts = nn.ModuleList([Expert([embedding_size] + experts_units + [mmoe_hidden_dim]) for _ in range(n_expert)])
        # gates
        self.gates = nn.ModuleList([nn.Linear(embedding_size, n_expert) for _ in range(task_num)])
        for gate in self.gates:
            gate.weight.data.uniform_(0, 0.005)
        # task Tower
        hid_dim = [mmoe_hidden_dim] + hidden_units
        self.task_towers = nn.ModuleList([DNNTower(hid_dim) for _ in range(task_num)])

    def forward(self, x1, x2):
        x1 = x1 + x1.new_tensor(self.offsets)
        embeds = self.embeddings(x1)
        hidden = embeds.reshape(x1.shape[0], -1)
        hidden = torch.cat((hidden, x2), 1).to(torch.float32)
        experts_out = [expert(hidden) for expert in self.experts]
        experts_out = torch.stack(experts_out, dim=-1)
        gates_out = [gate.to(device)(hidden).to(device) for gate in self.gates]
        gates_out = torch.stack(gates_out)
        outs = torch.einsum("tbe, bme -> tbm", gates_out, experts_out)
        # task tower
        task_outs = [self.task_towers[task](outs[task]) for task in range(self.task_num)]
        return task_outs


def data_generate():
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_50k']
    categorical_columns = ['workclass', 'education', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    continuous_columns = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    train_df = pd.read_csv(data_path+'adult.data', delimiter=',',
                           header=None, index_col=None, names=column_names)
    test_df = pd.read_csv(data_path+'adult.test', delimiter=',',
                          header=None, index_col=None, names=column_names)
    train_df['tag'] = 1
    test_df['tag'] = 0
    test_df.dropna(inplace=True)
    test_df['income_50k'] = test_df['income_50k'].apply(lambda x: x[:-1])
    data = pd.concat([train_df, test_df])
    data.dropna(inplace=True)
    data['income_50k'] = data['income_50k'].apply(lambda x: 0 if x == ' <=50K' else 1)
    data['marital_status'] = data['marital_status'].apply(lambda x: 0 if x == ' Never-married' else 1)

    le = LabelEncoder()
    mm = MinMaxScaler()
    concrete_features = []
    for col in categorical_columns:
        data[col] = le.fit_transform(data[col])
        concrete_features.append(len(data[col].unique()))
    continuous_num = 0
    for col in continuous_columns:
        data[col] = mm.fit_transform(data[[col]]).reshape(-1)
        continuous_num += 1

    train_data, test_data = data[data['tag'] == 1], data[data['tag'] == 0]
    train_data.drop('tag', axis=1, inplace=True)
    test_data.drop('tag', axis=1, inplace=True)
    return train_data, test_data, concrete_features, continuous_num


if __name__ == "__main__":
    train_data, test_data, concrete_features, continuous_num = data_generate()
    train_dataset = CensusDataset(train_data)
    test_dataset = CensusDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    mmoe = MMOE(concrete_features, continuous_num, emb_dim=EMBEDDING_DIM,
                hidden_units=HIDDEN_UNITS, experts_units=EXPERT_UNITS).to(device)
    optimizer = torch.optim.Adam(mmoe.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    loss_func = nn.BCEWithLogitsLoss().to(device)

    # train
    train_loss_list = []
    test_income_auc_list = []
    test_marry_auc_list = []
    for i in range(EPOCH):
        total_loss, total_len = 0, 0
        for x1, x2, y1, y2 in train_dataloader:
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device).float(), y2.to(device).float()
            predict = mmoe(x1, x2)
            predict[0] = predict[0].squeeze()
            predict[1] = predict[1].squeeze()
            loss_1 = loss_func(predict[0], y1)
            loss_2 = loss_func(predict[1], y2)
            loss = loss_1 + loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            total_len += 1
        train_loss = total_loss / total_len
        train_loss_list.append(train_loss)

        mmoe.eval()
        income_label = []
        marry_label = []
        income_predict = []
        marry_predict = []
        for x1, x2, y1, y2 in test_dataloader:
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device).type(torch.float32), y2.to(device).type(torch.float32)
            predict = mmoe(x1, x2)
            predict[0] = predict[0].squeeze()
            predict[1] = predict[1].squeeze()
            income_label.extend(y1.tolist())
            marry_label.extend(y2.tolist())
            income_predict += predict[0]
            marry_predict += predict[1]
            loss_1 = loss_func(predict[0], y1)
            loss_2 = loss_func(predict[1], y2)
            loss = loss_1 + loss_2
        income_auc = roc_auc_score(np.array(income_label), np.array(income_predict))
        marry_auc = roc_auc_score(np.array(marry_label), np.array(marry_predict))
        test_income_auc_list.append(income_auc)
        test_marry_auc_list.append(marry_auc)
        print("epoch {}, train loss is {:.4f}, income auc is {:.4f}, "
              "marry auc is {:.4f}".format(i + 1, train_loss, income_auc, marry_auc))

    print("max auc in test dataset: income - {:.4f}, marry - {:.4f}".
          format(max(test_income_auc_list), max(test_marry_auc_list)))
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(test_income_auc_list, label='income_auc')
    plt.plot(test_marry_auc_list, label='marry_auc')
    plt.legend()
    plt.show()
