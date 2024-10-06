import torch.nn as nn
import torch
from torch.distributions import Categorical
from torch_geometric.data import Data

from model import ActorGCN, CriticGCN, TransActor, GATActor, MLPActor
from recommender import Recommender
from recommender import train_model
from replay_buffer import ReplayBufferGNN
from utils.node_utils import get_edge_attr, get_edge_index
import random
import numpy as np

class MLPAgent:
    def __init__(self, args, learning_rate=1e-6, gamma=0.99, buffer_size=1000):
        self.cache_size = args.cache_size
        self.agent_name = "MLP"
        self.args = args
        self.feature_dim = args.feature_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = self.args.rl_batch_size
        self.replay_buffer = ReplayBufferGNN(buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = MLPActor(self.feature_dim, self.feature_dim).to(self.device)
        self.critic = CriticGCN(self.feature_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate, weight_decay=1e-7)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, weight_decay=1e-7)

        self.MSE_loss = nn.MSELoss()

    def get_action(self, data, eps=0.1):
        action_prob, rsu_embedding = self.actor(data)
        m = Categorical(action_prob[0])
        action = m.sample().item()
        if np.random.rand() < eps:
            action = random.randint(0, 1)
            return action, rsu_embedding
        return action, rsu_embedding

    def optimize_model(self, batch_size):
        if self.replay_buffer.length() < batch_size:
            return

        data_loader = self.replay_buffer.sample(batch_size)
        if data_loader:
            for batch_idx, data in enumerate(data_loader):
                node_feature = data[batch_idx].node_feature.to(self.device)
                reward = data[batch_idx].reward.to(self.device)
                next_state = data[batch_idx].next_state.to(self.device)
                terminal = data[batch_idx].terminal.to(self.device)

                x_current = Data(node_feature=node_feature, edge_index=data[batch_idx].edge_index)
                current_action_probs, _ = self.actor(x_current)

                current_value = self.critic(node_feature[0]).squeeze()

                next_value = self.critic(next_state).squeeze()

                target_value = reward + self.gamma * next_value * (1 - terminal)
                advantage = target_value - current_value
                advantage = advantage.unsqueeze(-1)

                critic_loss = (advantage.pow(2)).mean()

                action_log_probs = torch.log(current_action_probs)
                actor_loss = -(action_log_probs * advantage.detach()).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

class DDQNAgent:
    def __init__(self, node_feature_dim, hidden_dim, num_actions, lr=1e-3, gamma=0.99, buffer_size=10000, batch_size=64, tau=1e-3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau  # 用于软更新目标网络参数

        # 主网络和目标网络
        self.policy_net = DQN_GNN(node_feature_dim, hidden_dim, num_actions).to(self.device)
        self.target_net = DQN_GNN(node_feature_dim, hidden_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # 经验回放缓冲区
        self.memory = ReplayBufferGNN(buffer_size)

        # 初始化时间步
        self.steps_done = 0

    def select_action(self, state, epsilon):
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                # 选择具有最大Q值的动作
                action = q_values.argmax(dim=1).item()
        else:
            # 随机选择动作
            action = random.randrange(self.num_actions)
        return action

    def optimize_model(self):
        if self.memory.length() < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = DataLoader(transitions, batch_size=self.batch_size, shuffle=False)
        for data in batch:
            # 将数据移动到设备上
            state_batch = data.node_feature.to(self.device)
            action_batch = data.action.to(self.device)
            reward_batch = data.reward.to(self.device)
            next_state_batch = data.next_state.to(self.device)
            terminal_batch = data.terminal.to(self.device)
            edge_index = data.edge_index.to(self.device)

            # 计算当前状态的Q值
            state_action_values = self.policy_net(Data(node_feature=state_batch, edge_index=edge_index)).gather(1, action_batch.unsqueeze(1))

            # 计算下一个状态的最大Q值（使用目标网络）
            with torch.no_grad():
                next_state_q_values = self.policy_net(Data(node_feature=next_state_batch, edge_index=edge_index))
                next_actions = next_state_q_values.argmax(dim=1, keepdim=True)
                next_state_target_q_values = self.target_net(Data(node_feature=next_state_batch, edge_index=edge_index))
                next_state_values = next_state_target_q_values.gather(1, next_actions)

            # 计算预期的Q值
            expected_state_action_values = (next_state_values * self.gamma * (1 - terminal_batch.unsqueeze(1))) + reward_batch.unsqueeze(1)

            # 计算损失
            loss = self.loss_fn(state_action_values, expected_state_action_values)

            # 优化模型
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 软更新目标网络参数
            for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

class GCNAgent:
    """
    This class is responsible for the creation of the GCN agent.
    """

    def __init__(self, args, learning_rate=1e-6, gamma=0.99, buffer_size=1000):
        """

        :param cache_size: cache_size of current RSU
        :param learning_rate:
        :param gamma: discount factor
        :param buffer_size: size of the replay buffer TODO: hyperparameter tuning
        """
        self.cache_size = args.cache_size
        self.args = args
        self.agent_name = "GCN"
        self.feature_dim = args.feature_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = self.args.rl_batch_size
        self.replay_buffer = ReplayBufferGNN(buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorGCN(self.feature_dim, self.feature_dim).to(self.device)
        self.critic = CriticGCN(self.feature_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate, weight_decay=1e-7)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, weight_decay=1e-7)

        self.MSE_loss = nn.MSELoss()

    def get_action(self, data, eps=0.1):
        """
        Get the action for the current state
        :param edge_index:
        :param state: current state
        :param eps: epsilon-greedy
        :param edge_attr: edge attribute
        :return: action
        """
        action_prob, rsu_embedding = self.actor(data, True)
        m = Categorical(action_prob[0])
        action = m.sample().item()
        if np.random.rand() < eps:
            action = random.randint(0, 1)
            return action, rsu_embedding
        return action, rsu_embedding

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
                node_feature = data[batch_idx].node_feature.to(self.device)
                reward = data[batch_idx].reward.to(self.device)
                next_state = data[batch_idx].next_state.to(self.device)
                terminal = data[batch_idx].terminal.to(self.device)
                scores = data[batch_idx].scores.to(self.device)
                # 计算当前状态的动作概率和价值
                x_current = Data(node_feature=node_feature, edge_index=data[batch_idx].edge_index)
                current_action_probs, _ = self.actor(x_current)
                current_value = self.critic(node_feature[0]).squeeze()

                # 计算下一个状态的价值
                next_value = self.critic(next_state).squeeze()

                # 计算目标值和优势
                target_value = reward + self.gamma * next_value * (1 - terminal)
                advantage = target_value - current_value
                advantage = advantage.unsqueeze(-1)

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
                     , data, v2i_rate, graph_sampler):
    """
    Train the agent using the mini-batch training
    :param env: environment
    :param agent: agent
    :param max_episodes: maximum number of episodes
    :param max_steps: maximum number of steps
    :param batch_size: batch size
    :param data: original graph generated by sampler
    :param v2i_rate: vehicle to infrastructure rate
    :param vehicle_dis: vehicle distance
    :param graph_sampler: graph sampler
    :return episode_rewards, cache_efficiency_list, request_delay_list
    """

    episode_rewards = []
    cache_efficiency_list = []
    request_delay_list = []
    vehicle_request_num = []

    request_dataset = data.x.tolist()
    for i in range(len(request_dataset)):
        vehicle_request_num.append(len(request_dataset[i]))
    terminal = 0
    for episode in range(max_episodes):
        state = env.state
        episode_reward = 0
        print("____________", episode, " Started " + "__________")
        if episode == max_episodes - 1:
            terminal = 1
        for step in range(max_steps):
            graph = data
            edge_index = graph.edge_index.tolist()
            edge_attr = graph.edge_attr.tolist()
            recommender = Recommender(50, 50)
            for i in range(20):
                x = train_model(graph, recommender)
            request_dataset = x.tolist()
            action, rsu_embedding = agent.get_action(state, edge_index, edge_attr)
            # TODO add edge_index, edge_attr right here.
            next_state, reward, cache_efficiency, request_delay = env.step(action, rsu_embedding, request_dataset,
                                                                           v2i_rate, step)
            agent.replay_buffer.add(state, action, reward, terminal, next_state, edge_index, edge_attr)
            episode_reward += reward

            agent.optimize_model(batch_size)

            # # if len(agent.replay_buffer) > batch_size:
            # if agent.replay_buffer.length() % batch_size == 0:
            #     agent.optimize_model(batch_size)

            if step == max_steps - 1:
                episode_rewards.append(episode_reward)
                cache_efficiency_list.append(cache_efficiency)
                request_delay_list.append(request_delay)

    return episode_rewards, cache_efficiency_list, request_delay_list


class TransAgent:
    def __init__(self, args, learning_rate=1e-6, gamma=0.99, buffer_size=1000):
        self.cache_size = args.cache_size
        self.args = args
        self.feature_dim = args.feature_dim
        self.learning_rate = learning_rate
        self.agent_name = "Trans"
        self.gamma = gamma
        self.batch_size = self.args.rl_batch_size
        self.replay_buffer = ReplayBufferGNN(buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = TransActor(self.feature_dim, self.feature_dim).to(self.device)
        self.critic = CriticGCN(self.feature_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate, weight_decay=1e-7)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, weight_decay=1e-7)

        self.MSE_loss = nn.MSELoss()

    def get_action(self, data, eps=0.1):
        action_prob, rsu_embedding = self.actor(data, True)
        m = Categorical(action_prob[0])
        action = m.sample().item()
        if np.random.rand() < eps:
            action = random.randint(0, 1)
            return action, rsu_embedding
        return action, rsu_embedding

    def optimize_model(self, batch_size):
        if self.replay_buffer.length() < batch_size:
            return

        data_loader = self.replay_buffer.sample(batch_size)
        if data_loader:
            for batch_idx, data in enumerate(data_loader):
                node_feature = data[batch_idx].node_feature.to(self.device)
                reward = data[batch_idx].reward.to(self.device)
                next_state = data[batch_idx].next_state.to(self.device)
                terminal = data[batch_idx].terminal.to(self.device)

                x_current = Data(node_feature=node_feature, edge_index=data[batch_idx].edge_index)
                current_action_probs, _ = self.actor(x_current)
                current_value = self.critic(node_feature[0]).squeeze()

                next_value = self.critic(next_state).squeeze()

                target_value = reward + self.gamma * next_value * (1 - terminal)
                advantage = target_value - current_value
                advantage = advantage.unsqueeze(-1)

                critic_loss = (advantage.pow(2)).mean()

                action_log_probs = torch.log(current_action_probs)
                actor_loss = -(action_log_probs * advantage.detach()).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()


class GATAgent:
    def __init__(self, args, learning_rate=1e-6, gamma=0.99, buffer_size=1000):
        self.cache_size = args.cache_size
        self.args = args
        self.feature_dim = args.feature_dim
        self.learning_rate = learning_rate
        self.agent_name = "GAT"
        self.gamma = gamma
        self.batch_size = self.args.rl_batch_size
        self.replay_buffer = ReplayBufferGNN(buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = GATActor(self.feature_dim, self.feature_dim).to(self.device)
        self.critic = CriticGCN(self.feature_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate, weight_decay=1e-7)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, weight_decay=1e-7)

        self.MSE_loss = nn.MSELoss()

    def get_action(self, data, eps=0.1):
        action_prob, rsu_embedding = self.actor(data, True)
        m = Categorical(action_prob[0])
        action = m.sample().item()
        if np.random.rand() < eps:
            action = random.randint(0, 1)
            return action, rsu_embedding
        return action, rsu_embedding

    def optimize_model(self, batch_size):
        if self.replay_buffer.length() < batch_size:
            return

        data_loader = self.replay_buffer.sample(batch_size)
        if data_loader:
            for batch_idx, data in enumerate(data_loader):
                node_feature = data[batch_idx].node_feature.to(self.device)
                reward = data[batch_idx].reward.to(self.device)
                next_state = data[batch_idx].next_state.to(self.device)
                terminal = data[batch_idx].terminal.to(self.device)

                x_current = Data(node_feature=node_feature, edge_index=data[batch_idx].edge_index)
                current_action_probs, _ = self.actor(x_current)
                current_value = self.critic(node_feature[0]).squeeze()

                next_value = self.critic(next_state).squeeze()

                target_value = reward + self.gamma * next_value * (1 - terminal)
                advantage = target_value - current_value
                advantage = advantage.unsqueeze(-1)

                critic_loss = (advantage.pow(2)).mean()

                action_log_probs = torch.log(current_action_probs)
                actor_loss = -(action_log_probs * advantage.detach()).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()




