from torch_geometric.nn import GATConv, GCNConv, TransformerConv
import torch.nn as nn
import torch
import torch.nn.functional as F
class ActorGCN(nn.Module):
    def __init__(self, node_feature_dim=20, hidden_dim=10000, output_dim=2):
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
        x = self.gcn1(node_feature, edge_index)
        x = self.batch_norm(x)
        rsu_embedding = x[0].unsqueeze(0)
        print(rsu_embedding.shape)
        x = self.linear(x)
        x = self.relu(x)
        action_prob = self.softmax(x)

        return action_prob, rsu_embedding


class DQN_GNN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, num_actions):
        super(DQN_GNN, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, data):
        x, edge_index = data.node_feature, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x


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
    def __init__(self, hidden_dim=10000):
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
    def __init__(self, node_feature_dim=20, hidden_dim=10000, output_dim=2):
        super(GATActor, self).__init__()
        self.heads = 4
        # 使用PyTorch Geometric的GCN层
        self.gcn1 = GATConv(node_feature_dim, hidden_dim, heads=self.heads, concat=False)
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



