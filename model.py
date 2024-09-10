from torch_geometric.nn import GATConv, GCNConv, TransformerConv
import torch.nn as nn
import torch
import torch.nn.functional as F
class ActorGCN(nn.Module):
    def __init__(self, node_feature_dim=20, hidden_dim=500, output_dim=2):
        super(ActorGCN, self).__init__()
        self.heads = 4
        # 使用PyTorch Geometric的GCN层
        self.gcn1 = GCNConv(node_feature_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, current_state=True):
        node_feature, edge_index = x.node_feature, x.edge_index
        if current_state:
            x = self.gcn1(node_feature, edge_index)

        else:

            x = self.gcn1(node_feature, edge_index)
        x = self.batch_norm(x)
        rsu_embedding = x[0].unsqueeze(0)
        x = self.linear(x)
        x = self.relu(x)
        action_prob = self.softmax(x)

        return action_prob, rsu_embedding


class CriticGCN(nn.Module):
    def __init__(self, hidden_dim=500):
        super(CriticGCN, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
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
#             x = F.relu(x)
#             x = self.gcn2(x, edge_index)
#
#         x = self.model_sequence2(x)
#         return x

class GATActor(nn.Module):
    def __init__(self, node_feature_dim=20, hidden_dim=500, output_dim=2):
        super(GATActor, self).__init__()
        self.heads = 4
        # 使用PyTorch Geometric的GCN层
        self.gcn1 = GATConv(node_feature_dim, hidden_dim, heads=self.heads)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.linear = nn.Linear(hidden_dim * self.heads, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, current_state=True):
        node_feature, edge_index = x.node_feature, x.edge_index
        if current_state:
            x = self.gcn1(node_feature, edge_index)

        else:

            x = self.gcn1(node_feature, edge_index)
        x = self.batch_norm(x)
        rsu_embedding = x[0].unsqueeze(0)
        x = self.linear(x)
        x = self.relu(x)
        action_prob = self.softmax(x)

        return action_prob, rsu_embedding

class TransActor(nn.Module):
    def __init__(self, node_feature_dim=20, hidden_dim=500, output_dim=2):
        super(TransActor, self).__init__()
        self.heads = 4
        # 使用PyTorch Geometric的GCN层
        self.gcn1 = TransformerConv(node_feature_dim, hidden_dim, heads=self.heads)
        self.gcn2 = TransformerConv(hidden_dim, hidden_dim, heads=self.heads)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        self.linear = nn.Linear(hidden_dim * self.heads, output_dim)
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



