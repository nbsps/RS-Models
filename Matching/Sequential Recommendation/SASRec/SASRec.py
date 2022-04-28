import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader


data_path = '../../dataset/criteo/train.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
EMBEDDING_DIM = 64
LEARNING_RATE = 5e-3
REGULARIZATION = 1e-6
BATCH_SIZE = 1024
EPOCH = 30
SEQ_NUM = 50
DROPOUT_RATE = 0.5
NUM_BLOCKS = 3
NUM_HEADS = 1


# define ml seq dataset
class MlSeqDataset(Dataset):
    def __init__(self, user_historys, n_users):
        self.user_histotys = user_historys
        self.n_users = n_users

    def __getitem__(self, item):
        seq = self.user_histotys[item]
        return torch.tensor(seq[:-1]).to(device), torch.tensor(seq[1:]).to(device)

    def __len__(self):
        return n_users


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units):
        super(PointWiseFeedForward, self).__init__()

        self.ffn1 = nn.Linear(hidden_units, hidden_units)
        self.relu = nn.ReLU()
        self.ffn2 = nn.Linear(hidden_units, hidden_units)

        self.ffn1.weight.data.normal_(0, 0.05)
        self.ffn2.weight.data.normal_(0, 0.05)

    def forward(self, inputs):
        return self.ffn2(self.relu(self.ffn1(inputs)))


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, pad_idx, emb_dim=20):
        super(SASRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.pad_idx = pad_idx
        self.item_emb = nn.Embedding(self.item_num+1, emb_dim, padding_idx=self.pad_idx)
        self.pos_emb = nn.Embedding(SEQ_NUM, emb_dim)
        self.emb_dropout = nn.Dropout(p=DROPOUT_RATE)

        self.attention_layernorms = nn.ModuleList([nn.LayerNorm(emb_dim, eps=1e-8) for _ in range(NUM_BLOCKS)])
        self.attention_layers = nn.ModuleList([nn.MultiheadAttention(emb_dim, NUM_HEADS, DROPOUT_RATE) for _ in range(NUM_BLOCKS)])
        self.forward_layernorms = nn.ModuleList([nn.LayerNorm(emb_dim, eps=1e-8) for _ in range(NUM_BLOCKS)])
        self.forward_layers = nn.ModuleList([PointWiseFeedForward(emb_dim) for _ in range(NUM_BLOCKS)])

        self.last_layernorm = nn.LayerNorm(emb_dim, eps=1e-8)

        self.item_emb.weight.data.normal_(0, 0.05)
        self.pos_emb.weight.data.normal_(0, 0.05)

    def log2premb(self, log_seqs):
        # log_seq (batch_size, n_seq)
        seqs = self.item_emb(log_seqs)  # seqs (batch_size, n_seq, emb_d)
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.tensor(positions).to(device))
        seqs = self.emb_dropout(seqs)
        timeline_mask = log_seqs != self.pad_idx
        seqs *= timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=device))

        for i in range(NUM_BLOCKS):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            seqs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= timeline_mask.unsqueeze(-1)
        predict_emb = self.last_layernorm(seqs)
        return predict_emb

    def forward(self, prev_seqs, pos_seqs):
        predict_emb = self.log2premb(prev_seqs)
        pos_embs = self.item_emb(pos_seqs)
        pos_logits = (predict_emb * pos_embs).sum(dim=-1)
        return pos_logits

    def predict(self, log_seqs, item_indices):
        predict_emb = self.log2premb(log_seqs)
        final_feat = predict_emb[:, -1, :]
        item_embs = self.item_emb(item_indices)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


def evaluate(model):
    NDCG = 0.0
    HT = 0.0
    num = 0.0

    for u in range(0, n_users):
        seq, idx = user_history[u][:-1] + [test_history[u][0]], [test_history[u][1]] + user_history[u][1:]
        predictions = -model.predict(torch.tensor([seq]).to(device), torch.tensor([idx]).to(device))
        predictions = predictions[0]
        # score(predictions[0] ~ top k rank) -> 0(predictions.argsort()[k]=0) -> k(predictions.argsort().argsort()[0])
        rank = predictions.argsort().argsort()[0].item()
        num += 1
        if rank < 10:
            NDCG += 1/np.log2(rank + 2)
            HT += 1
    return NDCG/num, HT/num


if __name__ == "__main__":
    df = pd.read_csv('../../../dataset/ml-100k/u.data', header=None, sep="\t")
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True)
    n_users, n_items, _, _ = df.nunique()
    df['item_id'] = LabelEncoder().fit_transform(df['item_id'])
    df['user_id'] = LabelEncoder().fit_transform(df['user_id'])
    padding_id = n_items
    # user2item
    user_history = {}
    test_history = {}
    for uid in df.user_id.unique():
        history = df[df.user_id == uid].item_id.values.tolist()
        test_history[uid] = history[:2]
        history = history[-50:]
        user_history[uid] = [padding_id] * (50 - len(history)) + history

    train_dataset = MlSeqDataset(user_history, n_users)
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SASRec(n_users, n_items, padding_id).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
    loss_func = nn.BCEWithLogitsLoss().to(device)

    NDCG_list, HT_list = [], []
    for epoch in range(EPOCH):
        total_loss, total_len = 0, 0
        for prev, target in train_dl:
            avg_loss = 0
            optimizer.zero_grad()
            predict = model(prev, target)
            pos_labels = torch.ones(predict.shape, device=device)
            loss = loss_func(predict, pos_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss
            total_len += 1
        with torch.no_grad():
            model.eval()
            t_test = evaluate(model)
            NDCG_list.append(t_test[0])
            HT_list.append(t_test[1])
        print("epoch {}, train loss is {:.4f}, test (NDCG@10: {:.4f}, HR@10: {:.4f})".
              format(epoch, total_loss / total_len, t_test[0], t_test[1]))
    print("max NDCG@10 is {:.4f}, HR@10 is {:.4f}".format(min(NDCG_list), min(HT_list)))
