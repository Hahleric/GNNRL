import torch.nn as nn
import torch
from torch.distributions import Categorical
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

    def __init__(self, cache_size, feature_dim, batch_size, learning_rate=0.01, gamma=0.99, buffer_size=1000):
        """

        :param cache_size: cache_size of current RSU
        :param learning_rate:
        :param gamma: discount factor
        :param buffer_size: size of the replay buffer TODO: hyperparameter tuning
        """
        self.cache_size = cache_size
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = ReplayBufferGNN(buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorGCN(self.feature_dim).to(self.device)
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

        data = Data(state=state,edge_index=edge_index, edge_attr=edge_attr, num_nodes=(edge_index.shape[1] // 2) + 1)
        assert data.edge_index.max() < data.num_nodes
        action_prob = self.actor(data, True)
        m = Categorical(action_prob[0])
        action = m.sample().item()
        if np.random.rand() < eps:
            action = random.randint(0, 1)
            return action
        return action

    def optimize_model(self, batch_size):
        """
        Compute the loss for the current batch
        :return: loss
        """
        if self.replay_buffer.length() < batch_size:
            return

        data_loader = self.replay_buffer.sample(batch_size)
        if data_loader:
            for batch_idx, data in enumerate(data_loader):
                state = data.state.to(self.device)
                reward = data.reward.to(self.device)
                next_state = data.next_state.to(self.device)
                terminal = data.terminal.to(self.device)

                # 计算当前状态的动作概率和价值
                x_current = Data(state=state, edge_index=data.edge_index, edge_attr=data.edge_attr)
                current_action_probs = self.actor(x_current)
                current_value = self.critic(x_current).squeeze()

                # 计算下一个状态的价值
                x_next = Data(state=next_state, edge_index=data.edge_index, edge_attr=data.edge_attr)
                next_value = self.critic(x_next, False).squeeze()

                # 计算目标值和优势
                target_value = reward + self.gamma * next_value * (1 - terminal)
                advantage = target_value - current_value

                # 计算Critic的损失
                critic_loss = (advantage.pow(2)).mean()

                # 计算Actor的损失
                action_log_probs = torch.log(current_action_probs)
                actor_loss = -(action_log_probs * advantage.detach()).mean()

                # 优化Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # 优化Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size
                     , request_dataset, v2i_rate):
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
    edge_attr = get_edge_attr(edge_index, request_dataset)

    for i in range(len(request_dataset)):
        vehicle_request_num.append(len(request_dataset[i]))
    terminal = 0
    for episode in range(max_episodes):
        state, edge_index, remaining_content, node_features = env.reset()
        episode_reward = 0
        print("____________", episode," Started " + "__________")
        if episode == max_episodes - 1:
            terminal = 1
        for step in range(max_steps):

            # TODO modify replay buffer, state
            #  state should still be a list of cached movies instead of node features.
            #  2 ways: 1. use cached movie state, recommend_list concatenation to build node_feature
            #          2. keep using concatenation as state, only modify first element.(Not Good).
            #  action = agent.get_action(node_features, edge_index, edge_attr)
            action = agent.get_action(state, edge_index, edge_attr)
            # TODO add edge_index, edge_attr right here.
            next_state, reward, cache_efficiency, request_delay= env.step(action, request_dataset, v2i_rate, step)
            agent.replay_buffer.add(state, action, reward, terminal, next_state, edge_index, edge_attr)
            episode_reward += reward

            agent.optimize_model(batch_size)

            # if len(agent.replay_buffer) > batch_size:
            if agent.replay_buffer.length() % batch_size == 0:
                agent.update(batch_size)

            if step == max_steps - 1:
                episode_rewards.append(episode_reward)
                cache_efficiency_list.append(cache_efficiency)
                request_delay_list.append(request_delay)

    return episode_rewards, cache_efficiency_list, request_delay_list
