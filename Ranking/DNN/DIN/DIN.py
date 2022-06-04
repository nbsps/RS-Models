import os
import torch
import numpy as np
import pandas as pd
from torch import optim, nn
from sklearn import metrics
from matplotlib import pyplot as plt
from torch.nn import Embedding, BatchNorm1d
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

data_path = '../../../dataset/amazon/amazon-books-100k.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
EMBEDDING_DIM = 20
LEARNING_RATE = 1e-3
REGULARIZATION = 1e-6
BATCH_SIZE = 64
EPOCH = 100
MAX_LEN = 8


# Amazon Book dataset(原论文amazon eletronic, 效果不如原论文， 主复现)
class AmazonBooksDataset(Dataset):
    def __init__(self, x, y):
        super(AmazonBooksDataset, self).__init__()
        self.items = x[:, :MAX_LEN+1]
        self.cates = x[:, MAX_LEN+1:(MAX_LEN+1) * 2]
        self.y = y

    def __getitem__(self, item):
        return self.items[item], self.cates[item], self.y[item]

    def __len__(self):
        return len(self.items)


# Dice 损失函数
class Dice(nn.Module):
    def __init__(self, dim):
        super(Dice, self).__init__()
        self.batchnorm = BatchNorm1d(dim)
        self.alpha = nn.Parameter(torch.zeros((1,)), requires_grad=True)

    def forward(self, x):
        x = self.batchnorm(x)
        p = torch.sigmoid(x)
        return x.mul(p) + self.alpha * x.mul(1 - p)


# 激活节点
class ActivationUnit(nn.Module):
    def __init__(self, embed_dim):
        super(ActivationUnit, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * (embed_dim + 2), 64),
            Dice(MAX_LEN),
            nn.Linear(64, 32),
            Dice(MAX_LEN),
            nn.Linear(32, 1),
        )
        self.mlp[0].weight.data.normal_(0, 0.005)
        self.mlp[2].weight.data.normal_(0, 0.005)
        self.mlp[4].weight.data.normal_(0, 0.005)

    def forward(self, x):
        h = x[:, :-1]
        t = x[:, [-1] * h.shape[1]]
        # outer product
        p = torch.einsum("bfe, bfu -> bfeu", h, t).reshape(h.shape[0], h.shape[1], -1)
        att = self.mlp(torch.cat([h, p, t], dim=2))
        return att


# DIN
class DeepInterestNetwork(nn.Module):
    def __init__(self, item_dim, cate_dim, embed_dim=4):
        super(DeepInterestNetwork, self).__init__()
        # 商品 embedding
        self.itme_emb = Embedding(item_dim, embed_dim)
        # 商品属性（category） embedding
        self.cate_emb = Embedding(cate_dim, embed_dim)
        # 激活层
        self.attention = ActivationUnit(embed_dim * 2)
        # mlp
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, 128),
            Dice(128),
            nn.Linear(128, 64),
            Dice(64),
            nn.Linear(64, 1)
        )
        self.mlp[0].weight.data.normal_(0, 0.05)
        self.mlp[2].weight.data.normal_(0, 0.05)
        self.mlp[4].weight.data.normal_(0, 0.05)

    def forward(self, x1, x2):
        mask = (x1 > 0).float().unsqueeze(-1)  # 标记pad（0）
        his_embeddings = self.itme_emb(x1).mul(mask)
        his_cate_embeddings = self.cate_emb(x2).mul(mask)
        his_embeddings = torch.cat([his_cate_embeddings, his_embeddings], dim=-1)
        att = self.attention(his_embeddings)
        his_input = his_embeddings[:, :-1].mul(mask[:, :-1]).mul(att)
        user_interest = his_input.sum(dim=1)
        concated = torch.hstack([user_interest, his_embeddings[:, -1]])
        output = self.mlp(concated)
        output = torch.sigmoid(output)
        return output.squeeze()


if __name__ == "__main__":
    data_df = pd.read_csv(data_path)
    data_df['hist_item_list'] = data_df.apply(lambda x: x['hist_item_list'].split('|'), axis=1)
    data_df['hist_cate_list'] = data_df.apply(lambda x: x['hist_cate_list'].split('|'), axis=1)

    # cate id labelencode
    cate_list = list(data_df['cateID'])
    data_df.apply(lambda x: cate_list.extend(x['hist_cate_list']), axis=1)
    cate_set = set(cate_list + ['0'])
    cate_encode = LabelEncoder().fit(list(cate_set))
    cate_set = cate_encode.transform(list(cate_set))
    cate_dim = max(cate_set) + 1
    # item id labelencode
    item_list = list(data_df['itemID'])
    data_df.apply(lambda x: item_list.extend(x['hist_item_list']), axis=1)
    item_set = set(item_list + ['0'])
    item_encode = LabelEncoder().fit(list(item_set))
    item_set = item_encode.transform(list(item_set))
    item_dim = max(item_set) + 1

    item_col = ['hist_item_{}'.format(i) for i in range(MAX_LEN)]
    cate_col = ['hist_cate_{}'.format(i) for i in range(MAX_LEN)]

    def deal(x, col):
        if len(x) > MAX_LEN:
            return pd.Series(x[-MAX_LEN:], index=col)
        else:
            pad = MAX_LEN - len(x)
            x = ['0' for _ in range(pad)] + x
            return pd.Series(x, index=col)

    cate_df = data_df['hist_cate_list'].apply(lambda x: deal(x, cate_col)).join(data_df[['cateID']])\
        .apply(cate_encode.transform)
    item_df = data_df['hist_item_list'].apply(lambda x: deal(x, item_col)).join(data_df[['itemID']])\
        .apply(item_encode.transform).join(cate_df).join(data_df['label'])
    data = item_df.values

    train_X, test_X, train_y, test_y = train_test_split(data[:, :-1], data[:, -1], train_size=0.9, random_state=2022)
    train_dataset = AmazonBooksDataset(train_X, train_y)
    test_dataset = AmazonBooksDataset(test_X, test_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = DeepInterestNetwork(item_dim, cate_dim, EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    loss_func = nn.BCELoss().to(device)

    train_loss_list = []
    test_auc_list = []
    for epoch in range(EPOCH):
        model.train()
        total_loss, total_len = 0, 0
        labels, predicts = [], []
        for x1, x2, y in train_loader:
            optimizer.zero_grad()
            x1, x2, y = x1.type(torch.int).to(device), x2.type(torch.int).to(device), y.type(torch.float32).to(device)
            predict = model(x1, x2)
            loss = loss_func(predict, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total_len += len(y)
            labels.extend(y.tolist())
            predicts.extend(predict.tolist())
        train_auc = metrics.roc_auc_score(np.array(labels), np.array(predicts))
        train_loss = total_loss / total_len
        train_loss_list.append(train_loss)

        model.eval()
        labels, predicts = [], []
        with torch.no_grad():
            for x1, x2, y in test_loader:
                x1, x2, y = x1.type(torch.int).to(device), x2.type(torch.int).to(device), y.type(torch.float32).to(device)
                predict = model(x1, x2)
                labels.extend(y.tolist())
                predicts.extend(predict.tolist())
        auc = metrics.roc_auc_score(np.array(labels), np.array(predicts))
        test_auc_list.append(auc)
        print("epoch {}, train loss is {:.4f}, train auc is {:.4f} test auc is {:.4f}".format(epoch, train_loss, train_auc, auc))

    print("max auc in test dataset: {:.4f}".format(max(test_auc_list)))
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(test_auc_list, label='test_auc')
    plt.legend()
    plt.show()
