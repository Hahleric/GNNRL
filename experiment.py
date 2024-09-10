import itertools
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch_geometric.data import Data
import matplotlib.pyplot as plt

from utils.Cache import FIFOCache, LRUCache
from utils.Vehicle import Vehicle

from cv2x import Environ


class Experiment(object):
    def __init__(self, args, model, dataloader, environment, agent):
        self.test_dic = dataloader.test_dic
        self.test_items = dataloader.test_items
        self.history_csr = dataloader.train_csr
        self.fifo_cache = FIFOCache(args.cache_size)
        self.lru_cache = LRUCache(args.cache_size)
        self.fifo_efficiency = []
        self.lru_efficiency = []
        self.model = model
        self.args = args
        self.replace_num = 10
        self.environment = environment
        self.agent = agent
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        V2I_min = 100  # minimum required data rate for V2I Communication
        bandwidth = int(540000)
        bandwidth_mbs = int(1000000)
        COVERED_VEH = 20
        REQUESTED_MOVIES = 50
        BATCH_SIZE = 32
        n_veh = self.args.rl_batch_size
        env_v = Environ(n_veh, V2I_min, bandwidth, bandwidth_mbs)
        vehicle_dis = np.random.normal(100, 50, n_veh)
        self.v2i_rate, self.v2i_rate_mbs = env_v.Compute_Performance_Static(vehicle_dis)

    def replace_users(self, current_users):
        # 将 current_users 转换为列表
        current_users_list = list(current_users)

        # 随机选择 10 个用户进行替换
        replace_indices = random.sample(range(len(current_users_list)), self.replace_num)

        # 从数据集中随机选择 10 个新用户
        new_users = random.sample(list(self.test_dic.keys()), self.replace_num)

        # 替换旧用户
        for index, new_user in zip(replace_indices, new_users):
            current_users_list[index] = new_user

        return current_users_list

    def create_star_graph_edge_index(self, batch_size):
        # 中心节点索引为0
        center_node = 0
        # 创建边索引
        edge_index = []

        # 中心节点连接到其他所有节点
        for i in range(1, batch_size + 1):
            edge_index.append([center_node, i])
            edge_index.append([i, center_node])

        # 将边索引转换为tensor并转置
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        return edge_index

    def start(self):
        h = self.model.get_embedding()
        steps = 0
        episode = 0
        # current_users = random.sample(list(self.test_dic.keys()), self.args.batch_size)
        episode_rewards = []
        cache_efficiency_list = []
        request_delay_list = []
        vehicle_request_num = []
        covered_vehicles = []
        for i in range(self.args.rl_batch_size):
            vehicle = Vehicle(random.randint(0, 10), random.sample(list(self.test_dic.keys()), 1))
            covered_vehicles.append(vehicle)
        while episode < 1:
            state, _, _ = self.environment.reset()
            episode_reward = []
            terminal = 0
            steps = 0
            episode_cache_efficiencies = []
            print("____________", episode, " Started " + "__________")
            test_items = random.sample(self.test_items, 10000)
            while steps < 500:
                for i in covered_vehicles:
                    if i.time_stamp >= i.speed:
                        covered_vehicles.remove(i)
                        new_veh = Vehicle(random.randint(0, 5), random.sample(list(self.test_dic.keys()), 1))
                        covered_vehicles.append(new_veh)

                items_ready_to_cache = []
                scores = []
                request_dataset = set()
                for i in covered_vehicles:
                    vehicle_items = test_items[self.args.feature_dim * i.time_stamp: (i.time_stamp + 1) * self.args.feature_dim]
                    score = self.model.get_score(h, i.user_number)
                    mask = torch.tensor(self.history_csr[i.user_number].todense(), device=self.device).bool()
                    score[mask] = -float('inf')

                    _, recommended_items = torch.topk(score, k=self.args.k_list)
                    sorted_items = recommended_items.cpu()
                    score = self.model.get_score_by_user_item(h, i.user_number, vehicle_items)
                    score = score.squeeze(0)
                    scores.append(score.tolist())
                    itr = 0
                    j = 0
                    while itr < 10:
                        if j == len(sorted_items[0]):
                            items_ready_to_cache.append(sorted_items[0][random.randint(0, len(sorted_items) - 1)])
                            itr += 1
                        elif sorted_items[0][j] in items_ready_to_cache:
                            j += 1
                        else:
                            items_ready_to_cache.append(sorted_items[0][j])
                            j += 1
                            itr += 1

                    for k in vehicle_items:
                        request_dataset.add(k)
                request_dataset = list(request_dataset)

                for i in covered_vehicles:
                    i.time_stamp += 1
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
                scores = torch.tensor(scores, dtype=torch.float32).to(self.device)
                node_feature = torch.cat([state, scores], dim=0).to(self.device)
                edge_index = self.create_star_graph_edge_index(self.args.rl_batch_size).to(self.device)
                data = Data(node_feature=node_feature, edge_index=edge_index)
                action, rsu_embedding = self.agent.get_action(data)
                next_state, reward, cache_efficiency, request_delay = self.environment.step(
                    action,
                    rsu_embedding,
                    request_dataset,
                    self.v2i_rate,
                    steps,
                    items_ready_to_cache)
                self.agent.replay_buffer.add(node_feature, action, reward, terminal, next_state, edge_index, scores)
                episode_reward.append(reward)
                if steps > 0:
                    episode_cache_efficiencies.append(cache_efficiency)

                self.agent.optimize_model(self.args.rl_batch_size)
                steps += 1
            episode_rewards.append(episode_reward)
            cache_efficiency_list.append(episode_cache_efficiencies)
            episode += 1

        # 绘制每个 episode 最后 100 个 step 的平均 cache efficiency
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(cache_efficiency_list[-1])), cache_efficiency_list[-1], marker='o')
        plt.xlabel('Step')
        plt.ylabel('Average Cache Efficiency')
        plt.title('Average Cache Efficiency per Step in the Last Episode')
        plt.savefig('rewards')
        plt.grid(True)
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(episode_rewards[-1])), episode_rewards[-1], marker='o')
        plt.xlabel('Step')
        plt.ylabel('Rewards')
        plt.title('Rewards per Step in the Last Episode')
        plt.grid(True)
        plt.show()
        plt.savefig('rewards')

        return episode_rewards, cache_efficiency_list, request_delay_list

    def start_regular_with_recommender(self):
        h = self.model.get_embedding()
        steps = 0
        episode = 0
        # current_users = random.sample(list(self.test_dic.keys()), self.args.batch_size)
        episode_rewards = []
        cache_efficiency_list = []
        request_delay_list = []
        vehicle_request_num = []
        covered_vehicles = []
        for i in range(self.args.rl_batch_size):
            vehicle = Vehicle(random.randint(0, 10), random.sample(list(self.test_dic.keys()), 1))
            covered_vehicles.append(vehicle)
        while episode < 1:
            state, _ = self.environment.reset()
            episode_reward = []
            terminal = 0
            steps = 0
            episode_cache_efficiencies = []
            print("____________", episode, " Started " + "__________")
            test_items = random.sample(self.test_items, 10000)
            while steps < 50:
                for i in covered_vehicles:
                    if i.time_stamp >= i.speed:
                        covered_vehicles.remove(i)
                        new_veh = Vehicle(random.randint(0, 5), random.sample(list(self.test_dic.keys()), 1))
                        covered_vehicles.append(new_veh)

                items_ready_to_cache = []
                scores = []
                request_dataset = set()
                for i in covered_vehicles:
                    vehicle_items = test_items[self.args.feature_dim * i.time_stamp: (i.time_stamp + 1) * self.args.feature_dim]
                    score = self.model.get_score(h, i.user_number)
                    mask = torch.tensor(self.history_csr[i.user_number].todense(), device=self.device).bool()
                    score[mask] = -float('inf')

                    _, recommended_items = torch.topk(score, k=self.args.k_list)
                    sorted_items = recommended_items.cpu()
                    score = self.model.get_score_by_user_item(h, i.user_number, vehicle_items)
                    score = score.squeeze(0)
                    scores.append(score.tolist())
                    itr = 0
                    j = 0
                    while itr < 5:
                        if j == len(sorted_items[0]):
                            items_ready_to_cache.append(sorted_items[0][random.randint(0, len(sorted_items) - 1)])
                            itr += 1
                        elif sorted_items[0][j] in items_ready_to_cache:
                            j += 1
                        else:
                            items_ready_to_cache.append(sorted_items[0][j])
                            j += 1
                            itr += 1

                    for k in vehicle_items:
                        request_dataset.add(k)
                for item in items_ready_to_cache:
                    self.fifo_cache.put(item)
                    self.lru_cache.put(item)
                request_dataset = list(request_dataset)

                for i in covered_vehicles:
                    i.time_stamp += 1
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
                scores = torch.tensor(scores, dtype=torch.float32).to(self.device)
                node_feature = torch.cat([state, scores], dim=0).to(self.device)
                edge_index = self.create_star_graph_edge_index(self.args.rl_batch_size).to(self.device)
                data = Data(node_feature=node_feature, edge_index=edge_index)
                action, rsu_embedding = self.agent.get_action(data)
                next_state, reward, cache_efficiency, request_delay = self.environment.step(
                    action,
                    rsu_embedding,
                    request_dataset,
                    self.v2i_rate,
                    steps,
                    items_ready_to_cache)
                self.agent.replay_buffer.add(node_feature, action, reward, terminal, next_state, edge_index, scores)
                episode_reward.append(reward)
                if steps >= 0:
                    episode_cache_efficiencies.append(cache_efficiency)
                    fifo_hits = sum([1 for item in request_dataset if self.fifo_cache.get(item) is not None])
                    lru_hits = sum([1 for item in request_dataset if self.lru_cache.get(item) is not None])
                    self.fifo_efficiency.append(fifo_hits / len(request_dataset))
                    self.lru_efficiency.append(lru_hits / len(request_dataset))
                self.agent.optimize_model(self.args.rl_batch_size)
                steps += 1
            episode_rewards.append(episode_reward)
            cache_efficiency_list.append(episode_cache_efficiencies)
            episode += 1

            plt.figure(figsize=(10, 5))
            plt.plot(range(len(self.fifo_efficiency)), self.fifo_efficiency, marker='o', label='FIFO')
            plt.plot(range(len(self.lru_efficiency)), self.lru_efficiency, marker='o', label='LRU')
            plt.plot(range(len(cache_efficiency_list[-1])), cache_efficiency_list[-1], marker='o', label='GNNRL')
            plt.plot()
            plt.xlabel('Step')
            plt.ylabel('Cache Efficiency')
            plt.title('Cache Efficiency per Step in the Last Episode')
            plt.legend()
            plt.grid(True)
            plt.show()

        self.fifo_efficiency = []
        self.lru_efficiency = []
        return episode_rewards, cache_efficiency_list, request_delay_list
    def start_without_recommender(self):
        h = self.model.get_embedding()
        steps = 0
        episode = 0
        # current_users = random.sample(list(self.test_dic.keys()), self.args.batch_size)
        episode_rewards = []
        cache_efficiency_list = []
        request_delay_list = []
        vehicle_request_num = []
        covered_vehicles = []
        for i in range(self.args.rl_batch_size):
            vehicle = Vehicle(random.randint(0, 10), random.sample(list(self.test_dic.keys()), 1))
            covered_vehicles.append(vehicle)
        while episode < 1:
            state, _= self.environment.reset()
            episode_reward = []
            terminal = 0
            steps = 0
            episode_cache_efficiencies = []
            print("____________", episode, " Started " + "__________")
            test_items = random.sample(self.test_items, 10000)
            while steps < 1000:
                regular_items = random.sample(test_items, self.args.vehicle_num * 5)
                for i in regular_items:
                    self.fifo_cache.put(i)
                    self.lru_cache.put(i)
                for i in covered_vehicles:
                    if i.time_stamp >= i.speed:
                        covered_vehicles.remove(i)
                        new_veh = Vehicle(random.randint(0, 5), random.sample(list(self.test_dic.keys()), 1))
                        covered_vehicles.append(new_veh)

                items_ready_to_cache = []
                scores = []
                request_dataset = set()
                for i in covered_vehicles:
                    vehicle_items = test_items[self.args.feature_dim * i.time_stamp: (i.time_stamp + 1) * self.args.feature_dim]
                    score = self.model.get_score(h, i.user_number)
                    mask = torch.tensor(self.history_csr[i.user_number].todense(), device=self.device).bool()
                    score[mask] = -float('inf')

                    _, recommended_items = torch.topk(score, k=self.args.k_list)
                    sorted_items = recommended_items.cpu()
                    score = self.model.get_score_by_user_item(h, i.user_number, vehicle_items)
                    score = score.squeeze(0)
                    scores.append(score.tolist())
                    itr = 0
                    j = 0
                    while itr < 5:
                        if j == len(sorted_items[0]):
                            items_ready_to_cache.append(sorted_items[0][random.randint(0, len(sorted_items) - 1)])
                            itr += 1
                        elif sorted_items[0][j] in items_ready_to_cache:
                            j += 1
                        else:
                            items_ready_to_cache.append(sorted_items[0][j])
                            j += 1
                            itr += 1

                    for k in vehicle_items:
                        request_dataset.add(k)
                request_dataset = list(request_dataset)

                for i in covered_vehicles:
                    i.time_stamp += 1
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
                scores = torch.tensor(scores, dtype=torch.float32).to(self.device)
                node_feature = torch.cat([state, scores], dim=0).to(self.device)
                edge_index = self.create_star_graph_edge_index(self.args.rl_batch_size).to(self.device)
                data = Data(node_feature=node_feature, edge_index=edge_index)
                action, rsu_embedding = self.agent.get_action(data)
                next_state, reward, cache_efficiency, request_delay = self.environment.step(
                    action,
                    rsu_embedding,
                    request_dataset,
                    self.v2i_rate,
                    steps,
                    items_ready_to_cache)
                self.agent.replay_buffer.add(node_feature, action, reward, terminal, next_state, edge_index, scores)
                episode_reward.append(reward)
                if steps > 0:
                    episode_cache_efficiencies.append(cache_efficiency)
                    fifo_hits = sum([1 for item in request_dataset if self.fifo_cache.get(item) is not None])
                    lru_hits = sum([1 for item in request_dataset if self.lru_cache.get(item) is not None])
                    self.fifo_efficiency.append(fifo_hits / len(request_dataset))
                    self.lru_efficiency.append(lru_hits / len(request_dataset))
                self.agent.optimize_model(self.args.rl_batch_size)
                steps += 1
            episode_rewards.append(episode_reward)
            cache_efficiency_list.append(episode_cache_efficiencies)
            episode += 1

        # 绘制每个 episode 最后 100 个 step 的平均 cache efficiency
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.fifo_efficiency)), self.fifo_efficiency, marker='o', label='FIFO')
        plt.plot(range(len(self.lru_efficiency)), self.lru_efficiency, marker='o', label='LRU')
        plt.plot(range(len(cache_efficiency_list[-1])), cache_efficiency_list[-1], marker='o', label='GNNRL')
        plt.xlabel('Step')
        plt.ylabel('Cache Efficiency')
        plt.title('Cache Efficiency per Step in the Last Episode')
        plt.legend()
        plt.grid(True)
        plt.show()


        return episode_rewards, cache_efficiency_list, request_delay_list

