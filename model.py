from torch_geometric.nn import GATConv, GCNConv
import torch.nn as nn
import torch
import torch.nn.functional as F
class ActorGCN(nn.Module):
    def __init__(self, node_feature_dim=20, hidden_dim=500, output_dim=2):
        super(ActorGCN, self).__init__()
        self.heads = 4
        # 使用PyTorch Geometric的GCN层
        self.gcn1 = GCNConv(node_feature_dim, hidden_dim)
        self.model_sequence1 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, current_state=True):
        node_feature, edge_index = x.node_feature, x.edge_index
        if current_state:

            x = self.gcn1(node_feature, edge_index)

        else:
            # if the next state
            next_state = x.next_state
            next_state = torch.reshape(next_state, (-1, edge_attr.shape[1]))
            next_node_features = torch.cat((next_state, edge_attr), dim=0)
            x = self.gcn1(next_node_features, edge_index)
        x = self.model_sequence1(x)
        action_prob = self.softmax(x)
        rsu_embedding = x
        return action_prob, rsu_embedding


class CriticGCN(nn.Module):
    def __init__(self, node_feature_dim=20, hidden_dim=500):
        super(CriticGCN, self).__init__()
        self.gcn1 = GCNConv(node_feature_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        # 最终输出单个值，因此输出维度是1
        self.model_sequence2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, current_state=True):
        state,  edge_index, edge_attr = x.state, x.edge_index, x.edge_attr

        if current_state:
            state = torch.reshape(state, (-1, edge_attr.shape[1]))
            node_features = torch.cat((state, edge_attr), dim=0)
            x = self.gcn1(node_features, edge_index)
            x = F.relu(x)
            x = self.gcn2(x, edge_index)
        else:
            # if the next state
            next_state = x.state
            next_state = torch.reshape(next_state, (-1, edge_attr.shape[1]))
            next_node_features = torch.cat((next_state, edge_attr), dim=0)
            x = self.gcn1(next_node_features, edge_index)
            x = F.relu(x)
            x = self.gcn2(x, edge_index)

        x = self.model_sequence2(x)
        return x
