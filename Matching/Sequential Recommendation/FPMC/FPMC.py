import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

data_path = '../../../dataset/ml-100k/u.data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
EMBEDDING_DIM = 64
LEARNING_RATE = 1e-3
REGULARIZATION = 1e-6
BATCH_SIZE = 2048
EPOCH = 200


# define seq dataset of ml-100k
class MlSeqDataset(Dataset):
    def __init__(self, train_tuples, n_users, n_items):
        self.train_tuples = train_tuples
        self.n_users = n_users
        self.n_items = n_items
        self.n_tuples = len(train_tuples)

    def __getitem__(self, idx):
        uid, iidp, iidn = self.train_tuples[idx]
        # generate neg sample
        while True:
            iidneg = np.random.randint(0, self.n_items)
            if iidneg not in user_history[uid]:
                break
        sample = (uid, iidp, iidn, iidneg)
        return sample

    def __len__(self):
        return self.n_tuples


# define FPMC model
class FPMC(nn.Module):
    def __init__(self, n_users, n_items, k_UI=64, k_IL=64):
        super(FPMC, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.k_UI = k_UI
        self.k_IL = k_IL

        self.VIL = nn.Embedding(self.n_items, self.k_IL)
        self.VLI = nn.Embedding(self.n_items, self.k_IL)
        self.VUI = nn.Embedding(self.n_users, self.k_UI)
        self.VIU = nn.Embedding(self.n_items, self.k_UI)

        self.VIL.weight.data.uniform_(0, 0.05)
        self.VLI.weight.data.uniform_(0, 0.05)
        self.VUI.weight.data.uniform_(0, 0.05)
        self.VIU.weight.data.uniform_(0, 0.05)

    def forward(self, uid, basket_prev, iid):
        x_MF = torch.sum(self.VUI(uid) * self.VIU(iid), dim=1)
        x_FMC = torch.sum(self.VIL(iid) * self.VLI(basket_prev), dim=1)
        return x_MF + x_FMC

    def predict(self, u, b_tm1):
        rank_score = torch.matmul(self.VUI(u), self.VIU.weight.data.t()) + \
                     torch.mean(torch.matmul(self.VIL.weight.data, self.VLI(b_tm1).t()), dim=1)
        return rank_score


def recallN(model, data_list, N):
    list_recall = []
    for u, b_tm1, target_basket, _ in data_list:
        u, b_tm1, target_basket = u.to(device), b_tm1.to(device), target_basket.to(device)
        score = model.predict(u, b_tm1)
        idx = torch.topk(score, k=N).indices.tolist()
        for i in range(len(idx)):
            correct = len(set(idx[i]).intersection({target_basket[i].tolist()}))
            list_recall.append(correct)
    return np.mean(np.array(list_recall), axis=0)


if __name__ == "__main__":
    # load ml-100k dataset
    df = pd.read_csv(data_path, header=None, delimiter="\t")
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True)
    df['item_id'] = LabelEncoder().fit_transform(df['item_id'])
    df['user_id'] = LabelEncoder().fit_transform(df['user_id'])
    n_users, n_items, _, _ = df.nunique()
    # user2item
    user_history = {}
    for uid in df.user_id.unique():
        user_history[uid] = df[df.user_id == uid].item_id.values.tolist()
    # train test split
    train_history, test_history = [], []
    for uid, h in user_history.items():
        if len(h) < 5:
            continue
        for i in range(len(h) - 1):
            train_history.append((uid, h[i], h[i + 1]))
        # test_history.append((uid, h[i + 1], h[i + 2]))
        # test_history.append((uid, h[i + 2], h[i + 3]))
    train_history, test_history = train_test_split(train_history, train_size=0.7, random_state=2022)
    train_dataset = MlSeqDataset(train_history, n_users, n_items)
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = MlSeqDataset(test_history, n_users, n_items)
    test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # model SBPR_Loss optimizer
    model = FPMC(n_users, n_items, EMBEDDING_DIM, EMBEDDING_DIM).to(device)
    BPR_Loss = lambda p, n: torch.log(1 + torch.exp(-p + n)).sum()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)

    test_recall_list = []
    for epoch in range(EPOCH):
        train_loss = []
        for uid, iidp, iidn, iidneg in train_dl:
            uid, iidp, iidn, iidneg = uid.to(device), iidp.to(device), iidn.to(device), iidneg.to(device)
            avg_loss = 0
            optimizer.zero_grad()
            x_uit = model(uid, iidp, iidn)
            x_ujt = model(uid, iidp, iidneg)
            loss = BPR_Loss(x_uit, x_ujt)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        with torch.no_grad():
            model.eval()
            recall_test = recallN(model, test_dl, 10)
            test_recall_list.append(recall_test)
        print("epoch {}, train loss is {:.4f}, Recall@10 is {:.4f}".
              format(epoch, sum(train_loss) / len(train_history), recall_test))
    print("max Recall@10 in test dataset: {:.4f}".format(max(test_recall_list)))
