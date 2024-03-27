import torch.nn as nn
import torch
from torch_geometric.data import Data

from model import ActorGCN, CriticGCN
from replay_buffer import ReplayBufferGNN
from utils.node_utils import get_edge_attr, get_edge_index
import random
import numpy as np


class GCNAgent:
    """
    This class is responsible for the creation of the GCN agent.
    """

    def __init__(self, cache_size, batch_size, learning_rate=0.01, gamma=0.99, buffer_size=1000):
        """

        :param cache_size: cache_size of current RSU
        :param learning_rate:
        :param gamma: discount factor
        :param buffer_size: size of the replay buffer TODO: hyperparameter tuning
        """
        self.cache_size = cache_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = ReplayBufferGNN(buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorGCN().to(self.device)
        self.critic = CriticGCN().to(self.device)

        # TODO: hyperparameter tuning for weight_decay
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate, weight_decay=1e-7)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, weight_decay=1e-7)

        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, edge_index, edge_attr, eps=0.10):
        """
        Get the action for the current state
        :param edge_index:
        :param state: current state
        :param eps: epsilon-greedy
        :param edge_attr: edge attribute
        :return: action
        """
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(self.device)
        # state = torch.DoubleTensor(state).to(self.device)
        edge_index = torch.LongTensor(edge_index).to(self.device)


        data = Data(state=state, edge_index=edge_index, edge_attr=edge_attr, num_nodes=(edge_index.shape[1] // 2) + 1)
        assert data.edge_index.max() < data.num_nodes
        action = self.actor(data, True)

        if np.random.rand() < eps:
            action = random.randint(0, 1)
            return action
        return action

    def optimize_model(self):
        """
        Compute the loss for the current batch
        """
        if self.replay_buffer.length() < self.batch_size:
            return

        data_loader = self.replay_buffer.sample(self.batch_size)
        self.actor.eval()
        self.critic.eval()
        if data_loader:
            for batch_idx, data in enumerate(data_loader):
                state = data.state.to(self.device)
                action = data.action.to(self.device)
                reward = data.reward.to(self.device)
                next_state = data.next_state.to(self.device)
                terminal = data.terminal.to(self.device)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # Compute the loss for the actor
                actor_loss = -self.critic(state, action)
                actor_loss = actor_loss.mean()

                # Compute the loss for the critic
                next_action = self.actor(next_state)
                target = reward + self.gamma * self.critic(next_state, next_action) * (1 - terminal)
                critic_loss = self.MSE_loss(self.critic(state, action), target)
                critic_loss = critic_loss.mean()

                actor_loss.backward()
                critic_loss.backward()

                self.actor_optimizer.step()
                self.critic_optimizer.step()


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size
                     , request_dataset, v2i_rate, vehicle_dis):
    """
    Train the agent using the mini-batch training
    :param env: environment
    :param agent: agent
    :param max_episodes: maximum number of episodes
    :param max_steps: maximum number of steps
    :param batch_size: batch size
    :param request_dataset: request dataset
    :param v2i_rate: vehicle to infrastructure rate
    :param vehicle_dis: vehicle distance
    :return episode_rewards, cache_efficiency_list, request_delay_list
    """

    episode_rewards = []
    cache_efficiency_list = []
    request_delay_list = []
    vehicle_request_num = []
    vehicle_epoch = [i for i in range(1, request_dataset.shape[0] + 1)]
    edge_index = get_edge_index(len(vehicle_epoch))
    edge_attr = get_edge_attr(edge_index, vehicle_dis)

    for i in range(len(request_dataset)):
        vehicle_request_num.append(len(request_dataset[i]))

    for episode in range(max_episodes):
        state, edge_index, remaining_content = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state, edge_index, edge_attr)
            next_state, reward, cache_efficiency, request_delay = env.step(action, request_dataset, v2i_rate,
                                                                           vehicle_epoch,
                                                                           vehicle_request_num, step)
            agent.replay_buffer.add(state, action, reward, next_state)
            episode_reward += reward

            agent.optimize_model()

            # if len(agent.replay_buffer) > batch_size:
            if len(agent.replay_buffer) % batch_size == 0:
                agent.update(batch_size)

            if step == max_steps - 1:
                episode_rewards.append(episode_reward)
                cache_efficiency_list.append(cache_efficiency)
                request_delay_list.append(request_delay)

    return episode_rewards, cache_efficiency_list, request_delay_list
