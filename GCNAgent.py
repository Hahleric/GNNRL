import torch.nn as nn
import torch
from torch.distributions import Categorical
from torch_geometric.data import Data
import torch.distributions as distributions
from model import ActorGCN, CriticGCN, TransActor, ActorGAT, MLPActor, DQN_GNN
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
        scores, rsu_embedding = self.actor(data)
        # scores: shape [num_nodes, num_items]
        # 取 RSU 节点的评分
        rsu_scores = scores[0]  # 假设 RSU 节点的索引为 0
        if np.random.rand() < eps:
            # 随机选择
            action = random.sample(range(self.num_items), self.replace_num)
        else:
            # 根据评分选择前 replace_num 个文件
            _, topk_indices = torch.topk(rsu_scores, self.replace_num)
            action = topk_indices.tolist()
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


import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_geometric.data import Data
import math


# 假设 DQN_GNN 已经定义
# class DQN_GNN(nn.Module):
#     ...

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import random


class DDQNAgent:
    def __init__(self, args, lr=1e-4, gamma=0.75, tau=0.099):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_items = args.num_items  # Total number of items
        self.gamma = gamma
        self.agent_name = "DDQN"
        self.tau = tau  # For soft updating target network parameters
        self.num_actions_to_select = args.cache_size  # Number of files to cache (replace)

        # Initialize policy and target networks
        self.policy_net = DQN_GNN(args.feature_dim, args.hidden_dim, self.num_items).to(self.device)
        self.target_net = DQN_GNN(args.feature_dim, args.hidden_dim, self.num_items).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        # self.loss_fn = nn.MSELoss()

    def select_action(self, state, edge_index, items_ready_to_cache, epsilon=0.1):
        """
        Select multiple actions considering items_ready_to_cache
        """
        data = Data(node_feature=state, edge_index=edge_index)

        with torch.no_grad():
            q_values, rsu_embedding = self.policy_net(data, items_ready_to_cache)  # [len(items_ready_to_cache)]

        # Epsilon-greedy policy for multiple actions
        if random.random() > epsilon:
            # Select top K actions based on Q-values
            top_k = min(self.num_actions_to_select, len(items_ready_to_cache))
            top_k_indices = q_values.topk(top_k).indices.cpu().numpy()
            actions = [items_ready_to_cache[i] for i in top_k_indices]
        else:
            # Randomly select K actions from items_ready_to_cache
            top_k = min(self.num_actions_to_select, len(items_ready_to_cache))
            actions = random.sample(items_ready_to_cache, top_k)
        return actions, rsu_embedding

    def optimize_model(self, state, actions, reward, next_state, terminal, edge_index, items_ready_to_cache):
        """
        Use the current transition to optimize the model.
        """
        state = state.to(self.device)  # [num_nodes, feature_dim]
        next_state = next_state.to(self.device)
        edge_index = edge_index.to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)  # [1]
        terminal = torch.tensor([terminal], dtype=torch.float32).to(self.device)  # [1]

        # Get indices of actions in items_ready_to_cache
        action_indices = torch.tensor([items_ready_to_cache.index(a) for a in actions], dtype=torch.long).to(self.device)  # [K]

        # Current state's Q-values
        data = Data(node_feature=state, edge_index=edge_index)
        q_values, _ = self.policy_net(data, items_ready_to_cache)  # [len(items_ready_to_cache)]
        state_action_values = q_values[action_indices]  # [K]

        # Next state's Q-values (Double DQN)
        next_data = Data(node_feature=next_state, edge_index=edge_index)
        with torch.no_grad():
            next_q_values_policy, _ = self.policy_net(next_data, items_ready_to_cache)  # [len(items_ready_to_cache)]
            next_q_values_target, _ = self.target_net(next_data, items_ready_to_cache)  # [len(items_ready_to_cache)]

            # Select next actions based on policy net
            top_k = min(self.num_actions_to_select, len(items_ready_to_cache))
            next_action_indices = next_q_values_policy.topk(top_k).indices  # [K]
            next_state_values = next_q_values_target[next_action_indices]  # [K]

        # Compute expected Q values
        expected_state_action_values = reward + (self.gamma * next_state_values * (1 - terminal))

        loss = self.loss_fn(state_action_values, expected_state_action_values)

        # 计算损失
        print("Loss: ", loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network parameters
        self.soft_update_target_network()

    def soft_update_target_network(self):
        """
        Soft update target network parameters.
        """
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)



class GCNAgent:
    def __init__(self, args, learning_rate=1e-2, gamma=0.99, buffer_size=1000):
        self.cache_size = args.cache_size
        self.args = args
        self.agent_name = "GCN"
        self.feature_dim = args.feature_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = self.args.rl_batch_size
        self.replay_buffer = ReplayBufferGNN(buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replace_num = args.cache_size  # 每次要替换的文件数量

        self.actor = ActorGCN(self.feature_dim, self.feature_dim).to(self.device)
        self.critic = CriticGCN(self.feature_dim).to(self.device)

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=learning_rate, weight_decay=1e-7)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=learning_rate, weight_decay=1e-7)

        self.MSE_loss = nn.MSELoss()

    def get_action(self, data, items_ready_to_cache, eps=0.1):
        scores, rsu_embedding = self.actor(data, items_ready_to_cache)
        if np.random.rand() < eps:
            # 随机选择
            action_indices = random.sample(range(len(items_ready_to_cache)), self.replace_num)
        else:
            # 根据评分选择前 replace_num 个物品
            _, topk_indices = torch.topk(scores, self.replace_num)
            action_indices = topk_indices.tolist()
        # 将索引映射回物品 ID
        action = [items_ready_to_cache[i] for i in action_indices]
        return action, rsu_embedding

    def optimize_model(self, node_feature, action, reward, next_state, terminal, edge_index, items_ready_to_cache):
        # 将输入移动到指定设备
        node_feature = node_feature.to(self.device)

        # 将动作映射到索引
        try:
            action_indices = [items_ready_to_cache.index(a) for a in action]
        except ValueError as e:
            print(f"Action {e} not in items_ready_to_cache.")
            return  # 如果动作不在待缓存项中，跳过优化

        action_indices = torch.tensor(action_indices, dtype=torch.long).to(self.device)

        # 转换奖励和终端状态为张量
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        terminal = torch.tensor(terminal, dtype=torch.float32).to(self.device)

        # 将下一个状态和边索引移动到设备
        next_state = next_state.to(self.device)
        edge_index = edge_index.to(self.device)

        # 当前状态的图数据
        x_current = Data(node_feature=node_feature, edge_index=edge_index)

        # 获取当前状态下的评分和 RSU 嵌入
        current_scores, rsu_embedding = self.actor(x_current, items_ready_to_cache)

        # 计算动作的对数概率（假设使用的是概率策略）
        action_log_probs = torch.log_softmax(current_scores, dim=0)
        action_log_probs = action_log_probs[action_indices]

        # Critic 对当前状态的估计，使用 .detach() 以防止梯度回溯到 actor
        current_value = self.critic(rsu_embedding.detach()).squeeze()

        # 下一个状态的图数据
        x_next = Data(node_feature=next_state, edge_index=edge_index)

        with torch.no_grad():
            # 获取下一个状态的 RSU 嵌入
            _, next_rsu_embedding = self.actor(x_next, items_ready_to_cache)
            # Critic 对下一个状态的估计
            next_value = self.critic(next_rsu_embedding).squeeze()

        # 计算目标值
        target_value = reward + self.gamma * next_value * (1 - terminal)

        # 计算优势
        advantage = target_value - current_value

        # Critic 损失：当前值与目标值的均方误差
        critic_loss = self.MSE_loss(current_value, target_value.detach())

        # Actor 损失：策略梯度损失
        actor_loss = - (action_log_probs * advantage.detach()).mean()

        # 优化 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=False)  # 不需要保留计算图
        self.actor_optimizer.step()

        # 优化 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=False)  # 不需要保留计算图
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

    def optimize_model(self, node_feature, action, reward, next_state, terminal, edge_index, items_ready_to_cache):
        node_feature = node_feature.to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        terminal = torch.tensor(terminal).to(self.device)
        edge_index = edge_index.to(self.device)
        items_ready_to_cache = [int(item_id) for item_id in items_ready_to_cache]

        # 当前状态下的模型输出
        x_current = Data(node_feature=node_feature, edge_index=edge_index)

        current_scores, _ = self.actor(x_current, items_ready_to_cache)

        # 计算预测的 Q 值
        predicted_values = current_scores[action]

        # 下一个状态下的模型输出
        x_next = Data(node_feature=next_state, edge_index=edge_index)
        with torch.no_grad():
            next_scores, _ = self.actor(x_next, items_ready_to_cache)
            max_next_q_value = next_scores.max()

        # 计算目标 Q 值
        target_values = reward + self.gamma * max_next_q_value * (1 - terminal)

        # 计算损失
        loss = self.MSE_loss(predicted_values, target_values)

        # 优化模型
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()


class GATAgent:
    def __init__(self, args, learning_rate=1e-3, gamma=0.99):
        self.cache_size = args.cache_size
        self.args = args
        self.feature_dim = args.feature_dim
        self.learning_rate = learning_rate
        self.agent_name = "GAT"
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replace_num = args.replace_num  # Number of items to replace
        self.num_actions = args.cache_size  # Fixed action space size
        self.actor = ActorGAT(input_dim=self.feature_dim, hidden_dim=self.feature_dim, action_dim=self.num_actions).to(self.device)
        self.critic = CriticGCN(input_dim=self.feature_dim).to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate,
            weight_decay=1e-7
        )

        self.MSE_loss = nn.MSELoss()

    def get_action(self, data, items_ready_to_cache, current_cache):
        x, edge_index = data.node_feature, data.edge_index
        for i in range(len(x)):
            if torch.isnan(x[i]).any():
                print("Input x contains NaNs")
        action_scores, rsu_embedding = self.actor(x, edge_index)  # action_scores is of size self.num_actions

        # Debug statements to check for NaNs

        # Compute action probabilities using softmax
        action_probabilities = F.softmax(action_scores, dim=-1)

        # Check action_probabilities for NaNs

        # Create a categorical distribution
        action_distribution = torch.distributions.Categorical(probs=action_probabilities)
        # Sample actions
        action_indices = action_distribution.sample((self.replace_num,))
        action_indices = action_indices.tolist()
        print(len(action_indices), len(items_ready_to_cache))

        # Map action indices to items in items_ready_to_cache
        new_cache = current_cache.copy()
        for i in range(self.replace_num):
            new_cache[i] = items_ready_to_cache[action_indices[i]]

        return new_cache, rsu_embedding, action_indices, action_probabilities

    def optimize_model(self, rsu_embedding, action_indices, action_probabilities, reward, next_rsu_embedding, terminal):
        # Convert to tensors
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        terminal = torch.tensor([terminal], dtype=torch.float32, device=self.device)
        
        # Critic evaluation
        current_value = self.critic(rsu_embedding).squeeze()

        # Compute next value
        with torch.no_grad():
            next_value = self.critic(next_rsu_embedding).squeeze()
        # Compute target value
        target_value = reward + self.gamma * next_value * (1 - terminal)

        # Compute advantage
        advantage = target_value - current_value

        # Compute log probabilities of actions
        epsilon = 1e-10
        action_log_probs = torch.log(action_probabilities[action_indices] + epsilon)

        # Compute entropy for regularization
        entropy = -torch.sum(action_probabilities * torch.log(action_probabilities + 1e-10))
        # Actor loss
        entropy_coeff = 0.01  # Adjust coefficient as needed
        actor_loss = - (action_log_probs * advantage.detach()).mean() - entropy_coeff * entropy
        current_value = current_value.squeeze()
        target_value = target_value.squeeze()

        # Critic loss
        critic_loss = self.MSE_loss(current_value, target_value.detach())
        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

        # Then optimizer step
        self.optimizer.step()




