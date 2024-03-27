from torch_geometric.nn import GATConv, GCNConv
import torch.nn as nn
import torch
import torch.nn.functional as F
cache_size = 40
feature_size = 50
class ActorGCN(nn.Module):
    def __init__(self, node_feature_dim=feature_size, hidden_dim=1024, output_dim=2):
        super(ActorGCN, self).__init__()
        self.heads = 4
        # 使用PyTorch Geometric的GCN层
        self.gcn1 = GCNConv(node_feature_dim, hidden_dim)
        self.model_sequence1 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * self.heads),
            nn.Linear(node_feature_dim * self.heads, output_dim),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, current_state=True):
        state, edge_index, edge_attr = x.state, x.edge_index, x.edge_attr
        if current_state:
            state = torch.reshape(state, (state.shape[0], 1))
            # if the current state
            # TODO: 这里的input feature只有一个distance，完全不对，应该是某一辆车的目前推荐的电影
            x = self.gcn1(state, edge_index)
        else:
            # if the next state
            next_state = x.next_state
            next_state = torch.reshape(next_state, (next_state.shape[0], 1))
            x = self.gcn1(next_state, edge_index)
        x = self.model_sequence1(x)
        actor = self.softmax(x)
        return actor


class CriticGCN(nn.Module):
    def __init__(self, node_feature_dim=feature_size, hidden_dim=1024):
        super(CriticGCN, self).__init__()
        self.gcn1 = GCNConv(node_feature_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        # 最终输出单个值，因此输出维度是1
        self.model_sequence2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, current_state=True):
        state, edge_index, edge_attr = x.state, x.edge_index, x.edge_attr

        # state/next_state shape: [batch_size * num_nodes, time_sequence, state_dim]
        # -> [batch_size * num_nodes, time_sequence * state_dim]
        if current_state:
            state = torch.reshape(state, (state.shape[0], 1))
            # if the current state
            x = self.gcn1(state, edge_index)
        else:
            # if the next state
            next_state = x.next_state
            next_state = torch.reshape(next_state, (next_state.shape[0], 1))
            x = self.gcn1(next_state, edge_index)

        x = self.model_sequence1(x)
        return x
