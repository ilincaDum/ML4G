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

columns_name = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv("/content/u.data", sep="\t", names=columns_name)
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

def build_bipartite_edges(df_):
    edges = []
    for _, row in df_.iterrows():
        u_node = row['user_id_idx']
        i_node = row['item_id_idx'] + num_users
        edges.append((u_node, i_node))
        edges.append((i_node, u_node))
    return edges

edges = build_bipartite_edges(train_df)
adj_list = defaultdict(list)
for (src, dst) in edges:
    adj_list[src].append(dst)
item_pop_counter = Counter(train_df['item_id_idx'].tolist())
pop_dict = {i_idx + num_users: pop_count for i_idx, pop_count in item_pop_counter.items()}
for u_id in range(num_users):
    pop_dict[u_id] = 1

def random_walk_with_restart(start_node, adj_list, walk_length=10, restart_prob=0.3):
    path = [start_node]
    curr = start_node
    for _ in range(walk_length - 1):
        neighbors = adj_list.get(curr, [])
        if not neighbors:
            curr = start_node
            path.append(curr)
            continue
        if random.random() < restart_prob:
            curr = start_node
        else:
            curr = random.choice(neighbors)
        path.append(curr)
    return path

def precompute_importance_neighbors(num_nodes, adj_list, pop_dict, T=10, num_walks=3, walk_length=20, alpha=0.5):
    from collections import Counter
    precomp_neighbors = {}
    for node_id in range(num_nodes):
        visited = []
        for _ in range(num_walks):
            path = random_walk_with_restart(node_id, adj_list, walk_length, restart_prob=0.3)
            visited.extend(path)
        freq = Counter(visited)
        if node_id in freq:
            del freq[node_id]
        if len(freq) == 0:
            precomp_neighbors[node_id] = []
            continue
        raw_list = []
        for nbr, fcount in freq.items():
            pop_val = pop_dict.get(nbr, 1)
            raw_score = fcount * (pop_val ** (alpha * 0.5))
            raw_list.append((nbr, raw_score))
        raw_list.sort(key=lambda x: x[1], reverse=True)
        raw_list = raw_list[:T]
        sum_score = sum([x[1] for x in raw_list]) + 1e-9
        final_list = [(x[0], x[1] / sum_score) for x in raw_list]
        precomp_neighbors[node_id] = final_list
    return precomp_neighbors

num_nodes_total = num_users + num_items
offline_neighbors = precompute_importance_neighbors(num_nodes_total, adj_list, pop_dict, T=10, num_walks=3, walk_length=20, alpha=0.5)
print("Offline neighbor precomputation done. Example of node 0 neighbors:")
print(offline_neighbors[0])

T = 10
offline_nbr_ids = torch.full((num_nodes_total, T), fill_value=-1, dtype=torch.long)
offline_nbr_weights = torch.zeros((num_nodes_total, T), dtype=torch.float32)
for node in range(num_nodes_total):
    nbrs = offline_neighbors[node]
    for i, (nbr, w) in enumerate(nbrs):
        if i >= T:
            break
        offline_nbr_ids[node, i] = nbr
        offline_nbr_weights[node, i] = w
mask = (offline_nbr_ids == -1)
row_idx = torch.arange(num_nodes_total).unsqueeze(1).expand_as(offline_nbr_ids)
offline_nbr_ids[mask] = row_idx[mask]
offline_nbr_ids = offline_nbr_ids.to(device)
offline_nbr_weights = offline_nbr_weights.to(device)

class PinSageDataset(Dataset):
    def __init__(self, num_users, num_items, adj_list, train_df, num_samples=5):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.adj_list = adj_list
        self.num_samples = num_samples
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
        pos_node = None
        neg_node = None
        if center_node < self.num_users:
            pos_candidates = self.user_to_items[center_node]
            if len(pos_candidates) > 0:
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
            if len(pos_candidates) > 0:
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
        pos_node_val = pos_node if pos_node is not None else -1
        neg_node_val = neg_node if neg_node is not None else -1
        return {"center_node": center_node, "pos_node": pos_node_val, "neg_node": neg_node_val}

def my_collate_fn(batch_list):
    center_nodes = []
    pos_nodes = []
    neg_nodes = []
    for d in batch_list:
        center_nodes.append(d["center_node"])
        pos_nodes.append(d["pos_node"])
        neg_nodes.append(d["neg_node"])
    return {"center_node": torch.tensor(center_nodes, dtype=torch.long),
            "pos_node": torch.tensor(pos_nodes, dtype=torch.long),
            "neg_node": torch.tensor(neg_nodes, dtype=torch.long)}

class VectorizedPinSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Wn = nn.Linear(in_dim, out_dim)
        self.Ws = nn.Linear(in_dim, out_dim)
        self.Wc = nn.Linear(out_dim, out_dim)
    def forward(self, node_ids, global_emb_table, offline_nbr_ids, offline_nbr_weights):
        x_self = global_emb_table(node_ids)
        nbr_ids = offline_nbr_ids[node_ids]
        nbr_weights = offline_nbr_weights[node_ids]
        nbr_embs = global_emb_table(nbr_ids)
        weighted_nbr_embs = (nbr_embs * nbr_weights.unsqueeze(-1)).sum(dim=1)
        z_nbr = self.Wn(weighted_nbr_embs)
        z_self = self.Ws(x_self)
        z = F.relu(self.Wc(z_nbr + z_self))
        return z

class VectorizedPinSAGEModel(nn.Module):
    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([VectorizedPinSAGELayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
    def forward(self, node_ids, global_emb_table, offline_nbr_ids, offline_nbr_weights):
        x = global_emb_table(node_ids)
        for layer in self.layers:
            x = layer(node_ids, global_emb_table, offline_nbr_ids, offline_nbr_weights)
        return x

hidden_dim = 64
num_layers = 2
all_node_embeddings = nn.Embedding(num_nodes_total, hidden_dim).to(device)
nn.init.xavier_uniform_(all_node_embeddings.weight)
pinsage_model = VectorizedPinSAGEModel(hidden_dim, num_layers).to(device)

train_dataset = PinSageDataset(num_users, num_items, adj_list, train_df, num_samples=5)
BATCH_SIZE = 256
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=my_collate_fn)
optimizer = optim.Adam(list(pinsage_model.parameters()) + list(all_node_embeddings.parameters()), lr=1e-3)
margin_loss_fn = nn.MarginRankingLoss(margin=0.5)
EPOCHS = 10
print("Starting training...")
for epoch in range(EPOCHS):
    pinsage_model.train()
    total_loss = 0.0
    count = 0
    for batch_dict in train_loader:
        center_nodes = batch_dict["center_node"].to(device)
        pos_nodes = batch_dict["pos_node"].to(device)
        neg_nodes = batch_dict["neg_node"].to(device)
        valid_mask = (pos_nodes >= 0) & (neg_nodes >= 0)
        if valid_mask.sum() == 0:
            continue
        center_nodes = center_nodes[valid_mask]
        pos_nodes = pos_nodes[valid_mask]
        neg_nodes = neg_nodes[valid_mask]
        center_emb = pinsage_model(center_nodes, all_node_embeddings, offline_nbr_ids, offline_nbr_weights)
        pos_emb = pinsage_model(pos_nodes, all_node_embeddings, offline_nbr_ids, offline_nbr_weights)
        neg_emb = pinsage_model(neg_nodes, all_node_embeddings, offline_nbr_ids, offline_nbr_weights)
        pos_score = (center_emb * pos_emb).sum(dim=1)
        neg_score = (center_emb * neg_emb).sum(dim=1)
        target = torch.ones_like(pos_score)
        loss = margin_loss_fn(pos_score, neg_score, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
    avg_loss = total_loss / max(count, 1)
    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss = {avg_loss:.4f}")
print("Training Complete!")

@torch.no_grad()
def get_pinsage_embeddings(model, all_node_embeddings, offline_nbr_ids, offline_nbr_weights, num_nodes, batch_size=1024):
    model.eval()
    final_embs = torch.zeros((num_nodes, all_node_embeddings.embedding_dim), device=device)
    node_ids = torch.arange(num_nodes, device=device)
    for start in range(0, num_nodes, batch_size):
        batch_ids = node_ids[start:start+batch_size]
        updated = model(batch_ids, all_node_embeddings, offline_nbr_ids, offline_nbr_weights)
        final_embs[start:start+batch_ids.shape[0]] = updated
    return final_embs

final_embs = get_pinsage_embeddings(pinsage_model, all_node_embeddings, offline_nbr_ids, offline_nbr_weights, num_nodes_total, batch_size=256)
print("Final Embeddings computed!")
