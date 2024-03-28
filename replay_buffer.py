import collections
import numpy as np
import pickle
import random

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader

import logging


# Replay buffer
class ReplayBuffer:

    # create replay buffer of size N
    def __init__(self, N):
        self.buf = collections.deque(maxlen=N)

    # add: add a transition (s, a, r, s2, d)
    # add Data object directly
    def add(self, s, a, r, s2, d):
        # self.buf.append((s, a, r, s2, d))
        state, action, reward, next_state, terminal = s, a, r, s2, d

        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        terminal = torch.tensor(terminal, dtype=torch.long)
        data = Data(state=state, action=action, reward=reward, next_state=next_state, terminal=terminal)
        self.buf.append(data)

    # sample: return minibatch of size n
    def sample(self, batch_size):
        sample_data = random.sample(self.buf, batch_size)
        data_loader = DataLoader(sample_data, batch_size=len(sample_data), shuffle=False)
        return data_loader

    def clear(self):
        self.buf = collections.deque(maxlen=N)

    def length(self):
        return len(self.buf)


# Replay buffer
class ReplayBufferGNN(ReplayBuffer):

    # create replay buffer of size N
    def __init__(self, N):
        super().__init__(N)

    # add: add a transition (s, a, r, s2, d)
    # add Data object directly
    def add(self, s, a, r, s2, d, edge_index, edge_attr):
        # self.buf.append((s, a, r, s2, d))
        state, action, reward, next_state, terminal = s, a, r, s2, d

        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        terminal = torch.tensor(terminal, dtype=torch.long)

        edge_idnex = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.bool)
        num_nodes = state.shape[0]
        data = Data(state=state, action=action, reward=reward, next_state=next_state, terminal=terminal,
                    edge_index=edge_index, edge_attr=edge_attr, mask=mask, num_nodes=num_nodes)
        self.buf.append(data)
