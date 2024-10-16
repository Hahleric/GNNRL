import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.data import Data
import torch.optim as optim

from utils.GraphSampler import GraphSampler
from utils.data_preprocess_graph import get_graph


class Recommender(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4):
        super(Recommender, self).__init__()
        self.transformer_conv1 = TransformerConv(in_channels, out_channels, heads=heads, dropout=0.6, edge_dim=None,
                                                 beta=True)
        self.transformer_conv2 = TransformerConv(out_channels * heads, out_channels, heads=1, concat=False, dropout=0.6,
                                                 edge_dim=None, beta=True)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.transformer_conv1(x, edge_index, edge_weight))
        x = self.transformer_conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1), x


def train_model(data, model, epochs=20):
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.NLLLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out, x = model(data.x, data.edge_index, data.edge_weight)
        target = torch.randint(0, 1, (data.num_nodes,))  # Random target for example
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}: Loss {loss.item()}')
    return x


if __name__ == "__main__":
    dataset_path = 'ml-latest-small/ratings.csv'
    veh_num = 10
    sampler = GraphSampler(dataset_path, veh_num, 100)
    data = sampler.sample_movie()
    recommendation_size = 100
    print(data)
    model = Recommender(recommendation_size, recommendation_size)  # Set the number of features and classes as needed
    train_model(data, model)

