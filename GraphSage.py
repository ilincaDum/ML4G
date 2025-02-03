import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random
from collections import defaultdict, Counter
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = pd.read_csv("/content/ratings_small.csv")
df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'}, inplace=True)
df = df[df['rating'] >= 4].reset_index(drop=True)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=16)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

le_user = pp.LabelEncoder()
le_item = pp.LabelEncoder()
train_df['user_id_idx'] = le_user.fit_transform(train_df['user_id'].values)
train_df['item_id_idx'] = le_item.fit_transform(train_df['item_id'].values)
known_users = set(train_df['user_id'].unique())
known_items = set(train_df['item_id'].unique())
test_df = test_df[test_df['user_id'].isin(known_users) & test_df['item_id'].isin(known_items)]
test_df['user_id_idx'] = le_user.transform(test_df['user_id'].values)
test_df['item_id_idx'] = le_item.transform(test_df['item_id'].values)
num_users = train_df['user_id_idx'].nunique()
num_items = train_df['item_id_idx'].nunique()
print("num_users =", num_users, "| num_items =", num_items)

def build_bipartite_edges(df_, num_users):
    edges = []
    for _, row in df_.iterrows():
        u_node = row['user_id_idx']
        i_node = row['item_id_idx'] + num_users
        edges.append((u_node, i_node))
        edges.append((i_node, u_node))
    return edges

edges = build_bipartite_edges(train_df, num_users)
adj_list = defaultdict(list)
for (src, dst) in edges:
    adj_list[src].append(dst)
num_nodes_total = num_users + num_items
print("Total nodes (users+items):", num_nodes_total)

def precompute_graphsage_neighbors(num_nodes, adj_list, T=10):
    nbr_ids = torch.full((num_nodes, T), -1, dtype=torch.long)
    for node_id in range(num_nodes):
        neighbors = adj_list[node_id]
        if len(neighbors) == 0:
            nbr_ids[node_id, :] = node_id
        else:
            chosen = []
            for _ in range(T):
                chosen.append(random.choice(neighbors))
            nbr_ids[node_id, :] = torch.tensor(chosen, dtype=torch.long)
    mask = (nbr_ids == -1)
    row_idx = torch.arange(num_nodes).unsqueeze(1).expand_as(nbr_ids)
    nbr_ids[mask] = row_idx[mask]
    return nbr_ids

T = 5
graphsage_nbr_ids = precompute_graphsage_neighbors(num_nodes_total, adj_list, T=T)
graphsage_nbr_ids = graphsage_nbr_ids.to(device)
print("GraphSAGE neighbor matrix shape:", graphsage_nbr_ids.shape)

class GraphSAGEDataset(Dataset):
    def __init__(self, num_users, num_items, train_df):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.user_to_items = defaultdict(list)
        self.item_to_users = defaultdict(list)
        for _, row in train_df.iterrows():
            u = row['user_id_idx']
            i = row['item_id_idx']
            self.user_to_items[u].append(i)
            self.item_to_users[i].append(u)
        self.all_nodes = list(range(num_users + num_items))
    def __len__(self):
        return len(self.all_nodes)
    def __getitem__(self, idx):
        center_node = random.choice(self.all_nodes)
        if center_node < self.num_users:
            pos_candidates = self.user_to_items[center_node]
            if len(pos_candidates) == 0:
                return {"center_node": center_node, "pos_node": -1, "neg_node": -1}
            pos_item = random.choice(pos_candidates)
            pos_node = pos_item + self.num_users
            while True:
                if len(pos_candidates) == self.num_items:
                    neg_node = -1
                    break
                rand_item = random.randint(0, self.num_items - 1)
                if rand_item not in pos_candidates:
                    neg_node = rand_item + self.num_users
                    break
        else:
            i_id = center_node - self.num_users
            pos_candidates = self.item_to_users[i_id]
            if len(pos_candidates) == 0:
                return {"center_node": center_node, "pos_node": -1, "neg_node": -1}
            pos_user = random.choice(pos_candidates)
            pos_node = pos_user
            while True:
                if len(pos_candidates) == self.num_users:
                    neg_node = -1
                    break
                rand_user = random.randint(0, self.num_users - 1)
                if rand_user not in pos_candidates:
                    neg_node = rand_user
                    break
        return {"center_node": center_node, "pos_node": pos_node, "neg_node": neg_node}

def collate_fn(batch_list):
    center_nodes = []
    pos_nodes = []
    neg_nodes = []
    for d in batch_list:
        center_nodes.append(d["center_node"])
        pos_nodes.append(d["pos_node"])
        neg_nodes.append(d["neg_node"])
    return {
        "center_node": torch.tensor(center_nodes, dtype=torch.long),
        "pos_node": torch.tensor(pos_nodes, dtype=torch.long),
        "neg_node": torch.tensor(neg_nodes, dtype=torch.long)
    }

def gather_subgraph_nodes(batch_dict, graphsage_nbr_ids):
    center_nodes = batch_dict["center_node"].tolist()
    pos_nodes = batch_dict["pos_node"].tolist()
    neg_nodes = batch_dict["neg_node"].tolist()
    sub_nodes = set()
    for node in center_nodes + pos_nodes + neg_nodes:
        if node < 0:
            continue
        sub_nodes.add(node)
        neighbors_t = graphsage_nbr_ids[node].tolist()
        sub_nodes.update(neighbors_t)
    return list(sub_nodes)

class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_nbr = nn.Linear(in_dim, out_dim, bias=True)
        self.final_linear = nn.Linear(in_dim + out_dim, out_dim, bias=True)
    def forward(self, X_sub, sub_nodes, graphsage_nbr_ids, global_to_sub):
        in_dim = X_sub.shape[1]
        N_sub = X_sub.shape[0]
        node_expanded = graphsage_nbr_ids[sub_nodes]
        sub_nbr_indices = global_to_sub[node_expanded]
        self_mask = (sub_nbr_indices < 0)
        row_idx = torch.arange(N_sub, device=X_sub.device).unsqueeze(1).expand_as(sub_nbr_indices)
        sub_nbr_indices[self_mask] = row_idx[self_mask]
        nbr_embs = X_sub[sub_nbr_indices]
        nbr_mean = nbr_embs.mean(dim=1)
        h_nbr = self.linear_nbr(nbr_mean)
        h_nbr = F.relu(h_nbr)
        concat_vec = torch.cat([X_sub, h_nbr], dim=1)
        out = self.final_linear(concat_vec)
        out = F.relu(out)
        out = F.normalize(out, p=2, dim=1)
        return out

class GraphSAGEModel(nn.Module):
    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
    def forward(self, X_sub, sub_nodes, graphsage_nbr_ids):
        N_sub = X_sub.shape[0]
        device_ = X_sub.device
        global_to_sub = torch.full((num_nodes_total,), -1, dtype=torch.long, device=device_)
        global_to_sub[sub_nodes] = torch.arange(N_sub, device=device_)
        h = X_sub
        for layer in self.layers:
            h = layer(h, sub_nodes, graphsage_nbr_ids, global_to_sub)
        return h

hidden_dim = 64
num_layers = 2
all_node_embeddings = nn.Embedding(num_nodes_total, hidden_dim).to(device)
nn.init.xavier_uniform_(all_node_embeddings.weight)
sage_model = GraphSAGEModel(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
train_dataset = GraphSAGEDataset(num_users, num_items, train_df)
BATCH_SIZE = 256
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collate_fn)
optimizer = optim.Adam(list(sage_model.parameters()) + list(all_node_embeddings.parameters()), lr=1e-3)
margin_loss_fn = nn.MarginRankingLoss(margin=0.5)
EPOCHS = 10

for epoch in range(EPOCHS):
    sage_model.train()
    total_loss = 0.0
    batch_count = 0
    for batch in train_loader:
        center_nodes = batch["center_node"].to(device)
        pos_nodes = batch["pos_node"].to(device)
        neg_nodes = batch["neg_node"].to(device)
        valid_mask = (pos_nodes >= 0) & (neg_nodes >= 0)
        if valid_mask.sum() == 0:
            continue
        center_nodes = center_nodes[valid_mask]
        pos_nodes = pos_nodes[valid_mask]
        neg_nodes = neg_nodes[valid_mask]
        batch_dict = {"center_node": center_nodes, "pos_node": pos_nodes, "neg_node": neg_nodes}
        sub_nodes_list = gather_subgraph_nodes(batch_dict, graphsage_nbr_ids)
        sub_nodes_tensor = torch.tensor(sub_nodes_list, dtype=torch.long, device=device)
        X_sub = all_node_embeddings(sub_nodes_tensor)
        updated_sub = sage_model(X_sub, sub_nodes_tensor, graphsage_nbr_ids)
        global_to_sub = torch.full((num_nodes_total,), -1, dtype=torch.long, device=device)
        global_to_sub[sub_nodes_tensor] = torch.arange(len(sub_nodes_list), device=device)
        batch_loss = 0.0
        valid_count = 0
        for i in range(center_nodes.shape[0]):
            c = center_nodes[i].item()
            p = pos_nodes[i].item()
            n = neg_nodes[i].item()
            c_idx = global_to_sub[c]
            p_idx = global_to_sub[p]
            n_idx = global_to_sub[n]
            if c_idx < 0 or p_idx < 0 or n_idx < 0:
                continue
            c_emb = updated_sub[c_idx]
            p_emb = updated_sub[p_idx]
            n_emb = updated_sub[n_idx]
            pos_score = torch.dot(c_emb, p_emb)
            neg_score = torch.dot(c_emb, n_emb)
            y = torch.tensor([1.0], device=device)
            loss = margin_loss_fn(pos_score.unsqueeze(0), neg_score.unsqueeze(0), y)
            batch_loss += loss
            valid_count += 1
        if valid_count > 0:
            batch_loss = batch_loss / valid_count
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
        batch_count += 1
    avg_loss = total_loss / max(batch_count, 1)
    print(f"Epoch {epoch+1}/{EPOCHS} -- Loss: {avg_loss:.4f}")

print("Training Complete!")

@torch.no_grad()
def get_final_graphsage_embeddings(model, all_node_embeddings, graphsage_nbr_ids, num_nodes, batch_size=1024):
    model.eval()
    node_ids = np.arange(num_nodes)
    final_embs = {}
    for start in range(0, num_nodes, batch_size):
        chunk = node_ids[start:start+batch_size]
        sub_nodes = set(chunk)
        for nid in chunk:
            nbrs = graphsage_nbr_ids[nid].tolist()
            for nb in nbrs:
                sub_nodes.add(nb)
        sub_nodes = list(sub_nodes)
        sub_nodes_tensor = torch.tensor(sub_nodes, dtype=torch.long, device=device)
        X_sub = all_node_embeddings(sub_nodes_tensor)
        updated_sub = model(X_sub, sub_nodes_tensor, graphsage_nbr_ids)
        global_to_sub = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        global_to_sub[sub_nodes_tensor] = torch.arange(len(sub_nodes_tensor), device=device)
        for nid in chunk:
            idx = global_to_sub[nid].item()
            if idx >= 0:
                final_embs[nid] = updated_sub[idx].cpu()
    return final_embs

final_embeddings = get_final_graphsage_embeddings(sage_model, all_node_embeddings, graphsage_nbr_ids, num_nodes_total, batch_size=256)
print("Final GraphSAGE embeddings computed! #:", len(final_embeddings))

user_items_train = defaultdict(set)
for _, row in train_df.iterrows():
    u = int(row['user_id_idx'])
    i = int(row['item_id_idx'])
    user_items_train[u].add(i)

user_items_test = defaultdict(set)
for _, row in test_df.iterrows():
    u = int(row['user_id_idx'])
    i = int(row['item_id_idx'])
    user_items_test[u].add(i)

@torch.no_grad()
def evaluate_topk_hit_rate(final_embs, user_items_train, user_items_test, num_users, num_items, K=10):
    user_hit = 0
    user_count = 0
    item_vecs = []
    for i_id in range(num_items):
        it_node = i_id + num_users
        item_vecs.append(final_embs[it_node].unsqueeze(0))
    item_vecs = torch.cat(item_vecs, dim=0)
    user_ids_test = sorted(user_items_test.keys())
    for u_id in user_ids_test:
        test_itms = user_items_test[u_id]
        if not test_itms:
            continue
        user_count += 1
        u_vec = final_embs[u_id].unsqueeze(0)
        scores = torch.mv(item_vecs, u_vec.squeeze(0))
        train_items = user_items_train.get(u_id, set())
        for tr_i in train_items:
            scores[tr_i] = -999999.0
        _, topk_inds = torch.topk(scores, K)
        topk_inds = set(topk_inds.tolist())
        if len(topk_inds.intersection(test_itms)) > 0:
            user_hit += 1
    if user_count == 0:
        return 0.0
    return user_hit / user_count

@torch.no_grad()
def evaluate_mrr(final_embs, user_items_train, user_items_test, num_users, num_items):
    user_count = 0
    sum_rr = 0.0
    item_vecs = []
    for i_id in range(num_items):
        it_node = i_id + num_users
        item_vecs.append(final_embs[it_node].unsqueeze(0))
    item_vecs = torch.cat(item_vecs, dim=0)
    user_ids_test = sorted(user_items_test.keys())
    for u_id in user_ids_test:
        test_itms = user_items_test[u_id]
        if not test_itms:
            continue
        user_count += 1
        u_vec = final_embs[u_id].unsqueeze(0)
        scores = torch.mv(item_vecs, u_vec.squeeze(0))
        train_items = user_items_train.get(u_id, set())
        for tr_i in train_items:
            scores[tr_i] = -999999.0
        sorted_inds = torch.argsort(scores, descending=True)
        first_rank = None
        for rank, idx_ in enumerate(sorted_inds.tolist(), start=1):
            if idx_ in test_itms:
                first_rank = rank
                break
        if first_rank is not None:
            sum_rr += 1.0 / first_rank
    if user_count == 0:
        return 0.0
    return sum_rr / user_count

hit_rate_10 = evaluate_topk_hit_rate(
    final_embs=final_embeddings,
    user_items_train=user_items_train,
    user_items_test=user_items_test,
    num_users=num_users,
    num_items=num_items,
    K=10
)
mrr_score = evaluate_mrr(
    final_embs=final_embeddings,
    user_items_train=user_items_train,
    user_items_test=user_items_test,
    num_users=num_users,
    num_items=num_items
)

print(f"Hit@10: {hit_rate_10:.4f}")
print(f"MRR:    {mrr_score:.4f}")
