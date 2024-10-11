from torch_geometric.nn import GATConv, GCNConv, TransformerConv
import torch.nn as nn
import torch
import torch.nn.functional as F
class ActorGCN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim):
        super(ActorGCN, self).__init__()
        self.gcn1 = GCNConv(node_feature_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.item_embedding_layer = nn.Embedding(node_feature_dim, node_feature_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 可能需要增加一个线性层，将 RSU 嵌入转换到相同的维度
        self.rsu_projection = nn.Linear(hidden_dim, node_feature_dim)

    def forward(self, x, items_ready_to_cache):
        node_feature, edge_index = x.node_feature, x.edge_index
        x = self.gcn1(node_feature, edge_index)
        x = self.batch_norm(x)
        x = self.relu(x)

        rsu_embedding = x[0].unsqueeze(0)  # RSU 节点的嵌入表示，形状：[1, hidden_dim]
        rsu_embedding = self.rsu_projection(rsu_embedding)  # 转换到与物品嵌入相同的维度

        # 获取待缓存物品的嵌入表示
        items_embeddings = self.get_items_embeddings(items_ready_to_cache)  # 形状：[num_items, embed_dim]

        # 计算 RSU 与每个物品之间的相似度（例如，使用内积）
        scores = torch.matmul(items_embeddings, rsu_embedding.squeeze(0))  # 形状：[num_items]
        return scores, rsu_embedding

    def get_items_embeddings(self, items_ready_to_cache):

        item_id_to_index = {int(item_id): idx for idx, item_id in enumerate(items_ready_to_cache)}
        indices = [item_id_to_index[int(item_id)] for item_id in items_ready_to_cache]
        indices_tensor = torch.tensor(indices, dtype=torch.long).to(self.device)
        # 获取嵌入向量
        items_embeddings = self.item_embedding_layer(indices_tensor).to(self.device)
        return items_embeddings


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
class DQN_GNN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, num_items):
        super(DQN_GNN, self).__init__()

        # Define multiple GCN layers
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.batch_norm1 = nn.LayerNorm(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.batch_norm2 = nn.LayerNorm(hidden_dim)

        # Adding more convolution layers
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.batch_norm3 = nn.LayerNorm(hidden_dim)

        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.batch_norm4 = nn.LayerNorm(hidden_dim)

        # Adding fifth and sixth GCN layers

        # Embedding layer for items
        self.item_embeddings = nn.Embedding(num_items, hidden_dim)
        self.item_batch_norm = nn.BatchNorm1d(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, num_items),
            nn.ReLU(),
            nn.Linear(num_items, 1)
        )
        # Xavier initialization for the GCN layers
        torch.nn.init.xavier_uniform_(self.conv1.lin.weight)
        torch.nn.init.xavier_uniform_(self.conv2.lin.weight)
        torch.nn.init.xavier_uniform_(self.conv3.lin.weight)
        torch.nn.init.xavier_uniform_(self.conv4.lin.weight)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, data, items_ready_to_cache):
        x, edge_index = data.node_feature, data.edge_index
        batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        # First GCN layer with BatchNorm
        gelu = nn.GELU()
        x = gelu(self.batch_norm1(self.conv1(x, edge_index)))

        # Second GCN layer with BatchNorm
        x = gelu(self.batch_norm2(self.conv2(x, edge_index)))

        # Third GCN layer with BatchNorm
        x = gelu(self.batch_norm3(self.conv3(x, edge_index)))

        # Fourth GCN layer with BatchNorm
        x = F.leaky_relu(self.batch_norm4(self.conv4(x, edge_index)), negative_slope=0.01)

        rsu_embedding = x[0]  # Assume RSU node at index 0
        # Readout layer to get a graph-level embedding (global mean pool)
        graph_embedding = global_mean_pool(x, batch)  # shape [num_graphs, hidden_dim]

        # Assume RSU node is node 0 for each graph, extract its embedding
        # Get embeddings for items ready to cache with BatchNorm
        item_embs = self.get_items_embeddings(items_ready_to_cache)  # [num_items_ready, hidden_dim]
        item_embs = self.item_batch_norm(item_embs)

        # Compute Q-values as dot product between RSU embedding and item embeddings
        combined_features = torch.cat([item_embs, rsu_embedding.expand_as(item_embs)], dim=1)
        q_values = self.mlp(combined_features)  # [1, num_items_ready]
        q_values = q_values.T[0]
        return q_values, rsu_embedding  # Return Q-values and the graph-level embedding

    def get_items_embeddings(self, items_ready_to_cache):
        item_id_to_index = {int(item_id): idx for idx, item_id in enumerate(items_ready_to_cache)}
        indices = [item_id_to_index[int(item_id)] for item_id in items_ready_to_cache]
        indices_tensor = torch.tensor(indices, dtype=torch.long).to(self.device)

        # Get item embeddings
        items_embeddings = self.item_embeddings(indices_tensor).to(self.device)
        return items_embeddings



class MLPActor(nn.Module):
    def __init__(self, node_feature_dim=20, hidden_dim=10000, output_dim=2):
        super(MLPActor, self).__init__()
        self.fc1 = nn.Linear(node_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.node_feature
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        rsu_embedding = x[0].unsqueeze(0)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x, rsu_embedding
class CriticGCN(nn.Module):
    def __init__(self, input_dim):
        super(CriticGCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state_representation):
        x = F.relu(self.fc1(state_representation))
        x = self.fc2(x)
        return x




# class CriticGCN(nn.Module):
#     def __init__(self, node_feature_dim=20, hidden_dim=100):
#         super(CriticGCN, self).__init__()
#         self.gcn1 = GCNConv(node_feature_dim, hidden_dim)
#         self.gcn2 = GCNConv(hidden_dim, hidden_dim)
#         # 最终输出单个值，因此输出维度是1
#         self.model_sequence2 = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x, current_state=True):
#         node_feature, edge_index = x.node_feature, x.edge_index
#
#         if current_state:
#             x = self.gcn1(node_feature, edge_index)
#             x = F.relu(x)
#             x = self.gcn2(x, edge_index)
#         else:
#
#             x = self.gcn1(node_feature, edge_index)
#             x = F.relu(x)a
#             x = self.gcn2(x, edge_index)
#
#         x = self.model_sequence2(x)
#         return x

class ActorGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorGAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim)
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, action_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, edge_index):
        # Check for NaNs in input x and edge_index
        if torch.isnan(edge_index).any():
            print("edge_index contains NaNs")

        # Check for invalid edge indices
        num_nodes = x.size(0)
        if edge_index.max() >= num_nodes:
            print(f"Invalid edge_index: max index {edge_index.max()} exceeds number of nodes {num_nodes}")
        for i in range(len(x)):
            if torch.isnan(x[i]).any():
                print(f"x[{i}] contains NaNs")
        x = self.gat1(x, edge_index)
        if torch.isnan(x).any():
            print("After gat1, x contains NaNs")
        x = F.relu(x)
        if torch.isnan(x).any():
            print("After ELU, x contains NaNs")
        x = self.gat2(x, edge_index)
        if torch.isnan(x).any():
            print("After gat2, x contains NaNs")
        rsu_embedding = x[0]  # Assume RSU node at index 0
        if torch.isnan(rsu_embedding).any():
            print("rsu_embedding contains NaNs")
        action_scores = self.fc(rsu_embedding)  # Output action scores
        if torch.isnan(action_scores).any():
            print("action_scores contain NaNs")
        return action_scores, rsu_embedding


class TransActor(nn.Module):
    def __init__(self, node_feature_dim=20, hidden_dim=10000, output_dim=2):
        super(TransActor, self).__init__()
        self.heads = 4
        # 使用PyTorch Geometric的GCN层
        self.gcn1 = TransformerConv(node_feature_dim, hidden_dim, heads=self.heads, concat=False)
        self.gcn2 = TransformerConv(hidden_dim, hidden_dim, heads=self.heads, concat=False)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        self.linear = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, current_state=True):
        node_feature, edge_index = x.node_feature, x.edge_index
        x = self.gcn1(node_feature, edge_index)
        x = self.gcn2(x, edge_index)

        x = self.batch_norm(x)
        rsu_embedding = x[0].unsqueeze(0)
        x = self.linear(x)
        x = self.relu(x)
        action_prob = self.softmax(x)

        return action_prob, rsu_embedding



