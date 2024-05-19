# import numpy
# import os
#
# import regular_cache_enviroment
# from utils.GraphSampler import GraphSampler
#
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#
# import GCNAgent
# import environment
# import GCNAgent as Agent
# from GCNAgent import mini_batch_train
# from model import ActorGCN, CriticGCN
# from utils.data_preprocess import create_rating_matrix
# import numpy as np
# import torch
# from cv2x import V2Ichannels, Environ
#
# if __name__ == '__main__':
#     # c-v2x simulation parameters:
#
#     V2I_min = 100  # minimum required data rate for V2I Communication
#     bandwidth = int(540000)
#     bandwidth_mbs = int(1000000)
#     COVERED_VEH = 20
#     REQUESTED_MOVIES = 50
#     BATCH_SIZE = 32
#
#     data_set_path = 'ml-latest-small/ratings.csv'
#     user_ratings, top_num_movies, request_data, top_num_popular_movies \
#         = create_rating_matrix(data_set_path, REQUESTED_MOVIES)
#     n_veh = user_ratings.shape[0]
#     env_v = Environ(n_veh, V2I_min, bandwidth, bandwidth_mbs)
#     vehicle_dis = np.random.normal(100, 50, n_veh)
#     v2i_rate, v2i_rate_mbs = env_v.Compute_Performance_Static(vehicle_dis)
#
#     cache_size = 50
#     MAX_EPISODES = 30
#     # TODO currently, randomly select some users to train the model
#     # print request_data in a nice way
#     current_episode = 0
#     random_idx = np.random.choice(n_veh, COVERED_VEH, replace=True)
#     # sampled_requested_movies = np.array(request_data, shape=())[random_idx, :]
#     sampled_v2i_rate = v2i_rate[random_idx]
#     sampled_v2i_rate_mbs = v2i_rate_mbs[random_idx]
#     sampled_request_movies = top_num_movies[random_idx, :]
#     sampled_veh_dis = vehicle_dis[random_idx]
#
#     # 将所有采样用户的电影合并成一个一维数组
#     all_sampled_movies = sampled_request_movies.flatten()
#     # 计算每部电影出现的次数
#     unique_movies, counts = np.unique(all_sampled_movies, return_counts=True)
#
#     # 获取出现次数最多的50部电影的索引
#     top_50_indices = np.argsort(counts)[-REQUESTED_MOVIES:]
#
#     graph_sampler = GraphSampler(data_set_path, COVERED_VEH, REQUESTED_MOVIES)
#     sampled_top_50_movies = graph_sampler.get_recomended_movies()
#     data = graph_sampler.sample_movie()
#     sampled_request_movies = graph_sampler.get_node_features()
#
#     env = environment.Environment(cache_size, sampled_top_50_movies, sampled_request_movies)
#     agent = Agent.GCNAgent(cache_size, REQUESTED_MOVIES, 32)
#     episode_rewards, cache_efficiency, request_delay = mini_batch_train(env, agent, 30, 100,
#                                                                         16,
#                                                                         data,
#                                                                         sampled_v2i_rate,
#                                                                         graph_sampler,
#                                                                         )
#
#     env = regular_cache_enviroment.RegularEnvironment(cache_size, sampled_top_50_movies, sampled_request_movies)
#     env.calculate_base_policy()
import itertools
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch_geometric.data import Data
from utils.Vehicle import Vehicle

from cv2x import Environ


class Experiment(object):
    def __init__(self, args, model, dataloader, environment, agent):
        self.test_dic = dataloader.test_dic
        self.test_items = dataloader.test_items
        self.model = model
        self.args = args
        self.replace_num = 10
        self.environment = environment
        self.agent = agent
        V2I_min = 100  # minimum required data rate for V2I Communication
        bandwidth = int(540000)
        bandwidth_mbs = int(1000000)
        COVERED_VEH = 20
        REQUESTED_MOVIES = 50
        BATCH_SIZE = 32
        n_veh = self.args.batch_size
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
        for i in range(self.args.batch_size):
            vehicle = Vehicle(random.randint(0, 5), random.sample(list(self.test_dic.keys()), 1))
            covered_vehicles.append(vehicle)
        while episode < 100:
            state, _, _ = self.environment.reset()
            episode_reward = 0
            terminal = 0
            print("____________", episode, " Started " + "__________")
            test_items = random.sample(self.test_items, 10000)
            while steps < 1000:
                for i in covered_vehicles:
                    if i.time_stamp >= i.speed:
                        covered_vehicles.remove(i)
                        new_veh = Vehicle(random.randint(0, 5), random.sample(list(self.test_dic.keys()), 1))
                        covered_vehicles.append(new_veh)

                items_ready_to_cache = []
                scores = []
                for i in covered_vehicles:
                    vehicle_items = test_items[500*i.time_stamp: (i.time_stamp+1) * 500]
                    score = self.model.get_score_by_user_item(h, i.user_number, vehicle_items)
                    score = score.squeeze(0)
                    scores.append(score)
                    score, score_indices = torch.sort(score, descending=True)
                    sorted_items = [vehicle_items[i] for i in score_indices]
                    items_ready_to_cache.append(sorted_items[:5])
                items_ready_to_cache = list(itertools.chain.from_iterable(items_ready_to_cache))

                # scores = [score[:self.args.k_list] for score in scores]
                steps += 1
                state = torch.tensor(state, dtype=torch.float32)
                scores = torch.tensor(scores, dtype=torch.float32)
                node_feature = torch.cat([state, scores], dim=0)
                edge_index = self.create_star_graph_edge_index(self.args.batch_size)
                data = Data(x=node_feature, edge_index=edge_index)
                action, rsu_embedding = self.agent.get_action(data)
                next_state, reward, cache_efficiency, request_delay = self.environment.step(
                    action,
                    rsu_embedding,
                    request_dataset,
                    self.v2i_rate,
                    steps,
                    items_ready_to_cache)
                self.agent.replay_buffer.add(state, action, reward, terminal, next_state, edge_index, scores)
                episode_reward += reward

                self.agent.optimize_model(self.args.batch_size)

                if steps == 999:
                    episode_rewards.append(episode_reward)
                    cache_efficiency_list.append(cache_efficiency)
                    request_delay_list.append(request_delay)
        return episode_rewards, cache_efficiency_list, request_delay_list













