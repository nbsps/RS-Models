import os
import torch
import numpy as np
import collections
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

path = '../../../dataset/movielens'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
REGULARIZATION = 1e-6
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 20
BATCH_SIZE = 1024
MEMORY = 32
EPOCH = 50
HOP = 3


# dataset ripple set
class RippleSetDataset(Dataset):
    def __init__(self, data, ripple_set):
        self.data = data
        self.ripple_set = ripple_set

    def __getitem__(self, index):
        items = self.data[index, 1]
        labels = self.data[index, 2]
        hrt = self.ripple_set[self.data[index, 0]]
        return items, labels, hrt['h'], hrt['r'], hrt['t']

    def __len__(self):
        return len(self.data)


# model RippleNet
class RippleNet(nn.Module):
    def __init__(self, n_entity, n_relation):
        super(RippleNet, self).__init__()
        self.n_relation = n_relation
        self.dim = EMBEDDING_DIM
        self.n_entity = n_entity
        self.n_memory = MEMORY
        self.kge_weight = 0.01
        self.l2_weight = 1e-7
        self.n_hop = HOP
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim * self.dim)
        self.criterion = nn.BCELoss().to(device)

    def forward(self, items, labels, h, r, t):
        item_embeddings = self.entity_emb(items)
        hops_h_emb_list = []
        hops_r_emb_list = []
        hops_t_emb_list = []
        for i in range(self.n_hop):
            hops_h_emb_list.append(self.entity_emb(h[:, i, :]))
            hops_r_emb_list.append(self.relation_emb(r[:, i, :])\
                                   .view(-1, self.n_memory, self.dim, self.dim))
            hops_t_emb_list.append(self.entity_emb(t[:, i, :]))
        # calculate output
        o_list = []
        for hop in range(self.n_hop):
            h_ = torch.unsqueeze(hops_h_emb_list[hop], dim=-1)
            Rh = torch.squeeze(torch.matmul(hops_r_emb_list[hop], h_))
            v = torch.unsqueeze(item_embeddings, dim=-1)
            p = torch.squeeze(torch.matmul(Rh, v))
            p_norm = F.softmax(p, dim=-1)
            p_ = torch.unsqueeze(p_norm, dim=-1)
            o = (hops_t_emb_list[hop] * p_).sum(dim=1)
            o_list.append(o)
        scores = torch.sigmoid((item_embeddings * sum(o_list)).sum(dim=-1))
        # calculate loss
        loss = self.criterion(scores, labels.float())
        for hop in range(self.n_hop):
            h_expanded = torch.unsqueeze(hops_h_emb_list[hop], dim=2)
            t_expanded = torch.unsqueeze(hops_t_emb_list[hop], dim=3)
            hRt = torch.squeeze(torch.matmul(torch.matmul(h_expanded, hops_r_emb_list[hop]), t_expanded))
            loss -= self.kge_weight * torch.sigmoid(hRt).mean()
        for hop in range(self.n_hop):
            loss += self.l2_weight * (hops_h_emb_list[hop] * hops_h_emb_list[hop]).sum()
            loss += self.l2_weight * (hops_t_emb_list[hop] * hops_t_emb_list[hop]).sum()
            loss += self.l2_weight * (hops_r_emb_list[hop] * hops_r_emb_list[hop]).sum()
        return {"scores": scores, "loss": loss}


if __name__ == '__main__':
    ratings = pd.read_csv(path + '/ratings_final.csv', sep='\t', header=None)
    ratings.columns = ['user', 'item', 'label']
    train_ratings, test_ratings = train_test_split(ratings, train_size=0.7, random_state=2022)

    # set user history
    train_user_history = {}
    for i in train_ratings.user.unique():
        train_user_history[i] = train_ratings[train_ratings.user == i].item.values.tolist()
    test_user_history = {}
    for i in test_ratings.user.unique():
        test_user_history[i] = test_ratings[test_ratings.user == i].item.values.tolist()
    print("user history finished!")

    kg = pd.read_csv(path + '/kg_final.csv', sep='\t', header=None)
    n_entity = len(set(kg.iloc[:, 0]) | set(kg.iloc[:, 2]))
    n_relation = len(set(kg.iloc[:, 1]))

    # set knowledge graph dict
    kg_dict = collections.defaultdict(list)
    for h, r, t in kg.values:
        kg_dict[h].append((r, t))
    print("user history knowledge graph finished!")

    # set ripple set
    def deal_ripple_set(user_history, ripple_set):
        for user in user_history:
            ripple_set[user]['h'] = []
            ripple_set[user]['r'] = []
            ripple_set[user]['t'] = []
            for hop in range(HOP):
                hop_h, hop_r, hop_t = [], [], []

                if hop == 0:
                    tails_of_last_hop = user_history[user]
                else:
                    tails_of_last_hop = set(ripple_set[user]['t'][-1])

                for entity in tails_of_last_hop:
                    for relation, tail in kg_dict[entity]:
                        hop_h.append(entity)
                        hop_r.append(relation)
                        hop_t.append(tail)

                replace = len(hop_h) < MEMORY
                indices = np.random.choice(len(hop_h), size=MEMORY, replace=replace)
                ripple_set[user]['h'].append([hop_h[i] for i in indices])
                ripple_set[user]['r'].append([hop_r[i] for i in indices])
                ripple_set[user]['t'].append([hop_t[i] for i in indices])
            ripple_set[user]['h'] = torch.tensor(ripple_set[user]['h'])
            ripple_set[user]['r'] = torch.tensor(ripple_set[user]['r'])
            ripple_set[user]['t'] = torch.tensor(ripple_set[user]['t'])

    train_ripple_set = collections.defaultdict(dict)
    deal_ripple_set(train_user_history, train_ripple_set)
    test_ripple_set = collections.defaultdict(dict)
    deal_ripple_set(test_user_history, test_ripple_set)
    print("user ripple set of every hop finished!")

    train_dataset = RippleSetDataset(np.array(train_ratings), train_ripple_set)
    test_dataset = RippleSetDataset(np.array(test_ratings), test_ripple_set)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = RippleNet(n_entity, n_relation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=REGULARIZATION)

    print("start training!!!")
    train_loss_list, auc_list, acc_list = [], [], []
    for epoch in range(EPOCH):
        total_loss, total_len = 0, 0
        for items, labels, h, r, t in train_dataloader:
            items, labels, h, r, t = items.to(device), labels.to(device), h.to(device), r.to(device), t.to(device)
            return_dict = model(items, labels, h, r, t)
            loss = return_dict["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
            total_len += 1
        train_loss = total_loss / total_len
        train_loss_list.append(train_loss)

        model.eval()
        label_list, predicts = [], []
        with torch.no_grad():
            for items, labels, h, r, t in test_dataloader:
                items, labels, h, r, t = items.to(device), labels.to(device), h.to(device), r.to(device), t.to(device)
                return_dict = model(items, labels, h, r, t)
                label_list.extend(labels.tolist())
                predicts.extend(return_dict['scores'].tolist())
        auc = roc_auc_score(np.array(label_list), np.array(predicts))
        predictions = [1 if i >= 0.5 else 0 for i in predicts]
        acc = np.mean(np.equal(predictions, label_list))
        auc_list.append(auc)
        acc_list.append(acc)
        print("epoch {}, train loss is {:.4f}, test auc is {:.4f}, test acc is {:.4f}"
              .format(epoch, train_loss, auc, acc))

    print("max auc, acc in test dataset: {:.4f}, {:.4f}".format(max(auc_list), max(acc_list)))
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(auc_list, label='test_auc')
    plt.plot(acc_list, label='test_acc')
    plt.legend()
    plt.show()
