import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

data_path = "../../dataset/ml-100k/u.data"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
e_dim = 8
trail = 100
epochs = 100
batch_size = 4096
learning_rate = 1e-3
num_negative = 4
layers = [16, 32, 64]


# declare Click Dataset
class ClickDataset(Dataset):
    def __init__(self, uid, mid, click):
        self.uid = uid
        self.mid = mid
        self.click = click

    def __getitem__(self, index):
        return self.uid[index], self.mid[index], self.click[index]

    def __len__(self):
        return len(self.click)


# declare NeuralMF Model
class NeuralMF(nn.Module):

    def __init__(self, num_users, num_items, mf_dim, layers):
        super(NeuralMF, self).__init__()

        # MF embedding
        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=mf_dim)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=mf_dim)

        # MLP embedding
        self.MLP_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=layers[0] // 2)
        self.MLP_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=layers[0] // 2)

        # MLP
        self.dnn_network = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])
        self.linear = nn.Linear(layers[-1], mf_dim)

        # output
        self.linear2 = nn.Linear(2 * mf_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.MF_Embedding_User.weight.data.uniform_(0, 0.005)
        self.MF_Embedding_Item.weight.data.uniform_(0, 0.005)
        self.MLP_Embedding_User.weight.data.uniform_(0, 0.005)
        self.MLP_Embedding_Item.weight.data.uniform_(0, 0.005)
        self.linear.weight.data.uniform_(0, 0.005)
        self.linear2.weight.data.uniform_(0, 0.005)
        for i in self.dnn_network:
            i.weight.data.uniform_(0, 0.005)

    def forward(self, uid, mid):

        MF_Embedding_User = self.MF_Embedding_User(uid)
        MF_Embedding_Item = self.MF_Embedding_Item(mid)
        mf_vec = torch.mul(MF_Embedding_User, MF_Embedding_Item)
        MLP_Embedding_User = self.MLP_Embedding_User(uid)
        MLP_Embedding_Item = self.MLP_Embedding_Item(mid)
        x = torch.cat([MLP_Embedding_User, MLP_Embedding_Item], dim=-1)
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        mlp_vec = self.linear(x)
        vector = torch.cat([mf_vec, mlp_vec], dim=-1)
        linear = self.linear2(vector)
        output = self.sigmoid(linear)
        return output.squeeze()


# data preprocess
def predata(train, num_items, n_neg, user_item):
    user, item, labels = [], [], []
    for (u, i) in train:
        # positive sampling
        user.append(u)
        item.append(i)
        labels.append(1)
        # negative sampling
        for t in range(n_neg):
            j = np.random.randint(num_items)
            while j in np.array(user_item[u-1][1]):
                j = np.random.randint(num_items)
            user.append(u)
            item.append(j)
            labels.append(0)
    return [user, item], labels


if __name__ == "__main__":
    df = pd.read_csv(data_path, header=None, delimiter="\t")
    x, y = df.iloc[:, :2], df.iloc[:, 2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2022)
    # user_item 倒排表
    user_item = np.array(x_train.groupby(0)[1], dtype=object)
    x_train = x_train.values.tolist()
    train, label = predata(x_train, x.max(axis=0)[1], num_negative, user_item)

    train_dataset = ClickDataset(np.array(train[0]), np.array(train[1]), np.array(label).astype(np.float32))
    test_dataset = ClickDataset(np.array(x_test[0]), np.array(x_test[1]), np.array(np.ones(y_test.shape)).astype(np.float32))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataload = DataLoader(test_dataset, batch_size=batch_size)

    user_num, item_num = max(df[0]) + 1, max(df[1]) + 1
    model = NeuralMF(user_num, item_num, e_dim, layers).to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=trail)
    loss_func = nn.MSELoss().to(device)

    train_loss_list, test_mse_list = [], []
    for epoch in range(epochs):
        model.train()
        total_loss, total_len = 0, 0
        for x_u, x_i, y in train_dataloader:
            optim.zero_grad()
            x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
            predict = model(x_u, x_i)
            loss = loss_func(predict, y)
            loss.backward()
            optim.step()

            total_loss += loss.item() * len(y)
            total_len += len(y)
        train_loss = total_loss / total_len
        train_loss_list.append(train_loss)

        model.eval()
        labels, predicts = [], []
        with torch.no_grad():
            for x_u, x_i, y in test_dataload:
                x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
                predict = model(x_u, x_i)
                labels.extend(y.tolist())
                predicts.extend(predict.tolist())
        mse = mean_squared_error(np.array(labels), np.array(predicts))
        test_mse_list.append(mse)
        print("epoch {}, train loss is {}, val mse is {}".format(epoch, train_loss, mse))
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(test_mse_list, label='test_mse')
    plt.legend()
    plt.show()
