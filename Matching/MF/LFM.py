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
latent_k = 10


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
class LFMModel(nn.Module):
    def __init__(self, user_num, item_num, k=10):
        super(LFMModel, self).__init__()
        self.u_emb = nn.Embedding(user_num, k)
        self.i_emb = nn.Embedding(item_num, k)

        self.u_emb.weight.data.uniform_(0, 0.005)
        self.i_emb.weight.data.uniform_(0, 0.005)

    def forward(self, uid, mid):
        return (self.u_emb(uid) * self.i_emb(mid)).sum(1)


def main():
    df = pd.read_csv(data_path, header=None, delimiter="\t")
    x, y = df.iloc[:, :2], df.iloc[:, 2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2022)

    train_dataset = RatingDataset(np.array(x_train[0]), np.array(x_train[1]), np.array(y_train).astype(np.float32))
    test_dataset = RatingDataset(np.array(x_test[0]), np.array(x_test[1]), np.array(y_test).astype(np.float32))
    train_dataloader = DataLoader(train_dataset, batch_size=1024)
    test_dataload = DataLoader(test_dataset, batch_size=1024)

    user_num, item_num = max(df[0]) + 1, max(df[1]) + 1
    model = LFMModel(user_num, item_num, k=20).to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    loss_func = nn.MSELoss().to(device)

    test_mse_list = []
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
        print("epoch {}, train loss is {:.4f}, val mse is {:.4f}".format(epoch, train_loss, mse))
        writer.add_scalar("MF-MSE", mse, epoch)
        writer.add_scalar("MF-Loss", train_loss, epoch)
    print("min test mse is {:.4f}".format(min(test_mse_list)))
    writer.close()


if __name__ == "__main__":
    main()
