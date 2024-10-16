import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import numpy as np


def get_graph():
    ratings_path = 'ml-latest-small/ratings.csv'
    ratings_df = pd.read_csv(ratings_path, sep=',', header=0, names=['user_id', 'movie_id', 'rating', 'timestamp'])

    user_ids = ratings_df['user_id'].unique()
    movie_ids = ratings_df['movie_id'].unique()
    # Map IDs to consecutive indices starting from 0 for users and len(user_ids) for movies
    user_mapping = {user_id: i for i, user_id in enumerate(user_ids)}
    movie_mapping = {movie_id: i + len(user_ids) for i, movie_id in enumerate(movie_ids)}

    # Apply mappings to the DataFrame
    ratings_df['user_id'] = ratings_df['user_id'].map(user_mapping)
    ratings_df['movie_id'] = ratings_df['movie_id'].map(movie_mapping)

    # Create edge_index tensor from user IDs and movie IDs
    edge_index = torch.tensor([ratings_df['user_id'].values, ratings_df['movie_id'].values], dtype=torch.long)
    # Create edge attributes for weights (optional)
    edge_weight = torch.tensor(ratings_df['rating'].values, dtype=torch.float32)
    # Create a Data object

    num_nodes = len(user_ids) + len(movie_ids)


    # Make the graph undirected
    edge_index = to_undirected(edge_index)
    distance = np.random.rand(num_nodes, 1)
    node_features = torch.tensor(distance, dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight, num_nodes=num_nodes)
    data.num_nodes = num_nodes
    return data, distance



