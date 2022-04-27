import numpy as np
import torch.optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


data_path = "../../dataset/ml-100k/u.data"
device = torch.device("cuda")
writer = SummaryWriter("logs")
epochs = 100
latent_k = 20


# declare Ratings Dataset
class RatingDataset(Dataset):
    def __init__(self, uid, mid, rating):
        self.uid = uid
        self.mid = mid
        self.rating = rating

    def __getitem__(self, index):
        return self.uid[index], self.mid[index], self.rating[index]

    def __len__(self):
        return len(self.rating)


# declare MF model
class SVDppModel(nn.Module):
    def __init__(self, user_num, item_num, mean, k=10):
        super(SVDppModel, self).__init__()
        self.item_num = item_num
        self.u_emb = nn.Embedding(user_num, k)
        self.i_emb = nn.Embedding(item_num, k)
        self.u_bias = nn.Embedding(user_num, 1)
        self.i_bias = nn.Embedding(item_num, 1)
        self.y = nn.Embedding(item_num, k)

        self.u_emb.weight.data.uniform_(0, 0.005)
        self.i_emb.weight.data.uniform_(0, 0.005)
        self.u_bias.weight.data.uniform_(-0.01, 0.01)
        self.i_bias.weight.data.uniform_(-0.01, 0.01)
        self.y.weight.data.uniform_(0, 0.005)

        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)

    def forward(self, uid, mid, N):
        u_bias = self.u_bias(uid).squeeze()
        i_bias = self.i_bias(mid).squeeze()
        y = self.y(torch.arange(self.item_num).to(device))
        # print(N, N.matmul(y))
        x = N.matmul(y) / torch.sqrt(N.sum(1, keepdim=True))
        return ((self.u_emb(uid) + x) * self.i_emb(mid)).sum(1) + u_bias + i_bias + self.mean


def main():
    df = pd.read_csv(data_path, header=None, delimiter="\t")
    x, y = df.iloc[:, :2], df.iloc[:, 2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2022)

    train_dataset = RatingDataset(np.array(x_train[0]), np.array(x_train[1]), np.array(y_train).astype(np.float32))
    test_dataset = RatingDataset(np.array(x_test[0]), np.array(x_test[1]), np.array(y_test).astype(np.float32))
    train_dataloader = DataLoader(train_dataset, batch_size=1024)
    test_dataload = DataLoader(test_dataset, batch_size=1024)

    mean = df.iloc[:, 2].mean()
    user_num, item_num = max(df[0]) + 1, max(df[1]) + 1

    N = torch.zeros([user_num, item_num])
    user_item = np.array(df.groupby(0)[1], dtype=object)
    for i in range(user_num-1):
        N[i+1][np.array(user_item[i][1])] = 1

    model = SVDppModel(user_num, item_num, mean, k=20).to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    loss_func = nn.MSELoss().to(device)

    test_mse_list = []
    for epoch in range(epochs):
        model.train()
        total_loss, total_len = 0, 0
        for x_u, x_i, y in train_dataloader:
            optim.zero_grad()
            x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
            N_ = N[x_u].to(device)
            predict = model(x_u, x_i, N_)
            loss = loss_func(predict, y)
            loss.backward()
            optim.step()

            total_loss += loss.item() * len(y)
            total_len += len(y)
        train_loss = total_loss / total_len

        model.eval()
        labels, predicts = [], []
        with torch.no_grad():
            for x_u, x_i, y in test_dataload:
                x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
                N_ = N[x_u].to(device)
                predict = model(x_u, x_i, N_)
                labels.extend(y.tolist())
                predicts.extend(predict.tolist())
        mse = mean_squared_error(np.array(labels), np.array(predicts))
        test_mse_list.append(mse)
        print("epoch {}, train loss is {}, val mse is {}".format(epoch, train_loss, mse))
        writer.add_scalar("MF-MSE", mse, epoch)
        writer.add_scalar("MF-Loss", train_loss, epoch)
    print("min test mse is {:.4f}".format(min(test_mse_list)))
    writer.close()


if __name__ == "__main__":
    main()
