import random

import numpy as np
from utils.cache_utils import cache_hit_ratio, cache_hit_ratio2
from utils.node_utils import get_edge_index, get_edge_attr
import random
import math
import torch

import random
import torch
import numpy as np
import math
import torch.nn.init as init

class Cache:
    def __init__(self, file_id):
        self.file_id = file_id
        self.hits = 0


class Environment:
    def __init__(self, args, cache_size, test_items):
        """
        :param cache_size: cache size of the RSU, which is the only agent in our environment
        :param test_items: the sorted movie list by popularity, generated by recommender as well, it is
                            the sum of all movie rates, and sorted by the rate. Movies! not rates!
        """
        self.args = args
        self.cache_size = cache_size
        self.popular_files = test_items
        self.action_space = [0, 1]
        self.reward = 0
        self.cached_files = []
        self.init_n_veh = self.args.vehicle_num
        self.init_n_node = 1 + self.init_n_veh
        self.init_edge_index = get_edge_index(self.init_n_veh)
        self.edge_index = self.init_edge_index.copy()
        self.init_edge_index = torch.tensor(self.init_edge_index, dtype=torch.long).t()

        if len(self.popular_files) == self.cache_size:
            self.cached_files = [Cache(file_id) for file_id in self.popular_files]
        else:
            self.cached_files = [Cache(file_id) for file_id in random.sample(self.popular_files, self.cache_size)]

        self.init_cached_files = self.cached_files.copy()
        init_state = torch.empty((1, self.args.feature_dim))
        init.xavier_uniform_(init_state)
        self.init_state = init_state.numpy()
        self.state = self.init_state.copy()
        self.current_state = self.init_state.copy()

    def step(self, action, rsu_embedding, request_dataset, v2i_rate, print_step, items_ready_to_cache):
        # action 是一个包含待更换文件索引的列表
        if action:
            # 根据 action 更新缓存
            replace_items = action  # 待替换的文件列表
            replace_content = [Cache(file_id) for file_id in replace_items]
            # 替换缓存中的文件
            self.cached_files = replace_content
        # 更新状态为新的 RSU 嵌入表示
        self.current_state = self.state
        self.state = rsu_embedding
        # 计算缓存效率和奖励
        cache_efficiency, cached_items = cache_hit_ratio(request_dataset, self.cached_files)
        reward, request_delay = self.calculate_reward_and_delay(request_dataset, v2i_rate, cache_efficiency)
        if print_step % 50 == 0:
            print("---------------------------------------------")
            print('step {}: RSU1 cache_efficiency: {:.2f}'.format(print_step, cache_efficiency))
            print('step {}: request delay: {:.6f}'.format(print_step, request_delay))
            print('step {}: reward: {:.6f}'.format(print_step, reward))
            print("---------------------------------------------")
        return self.state, reward, cache_efficiency, request_delay, self.current_state

    # def calculate_reward_and_delay(self, request_dataset, v2i_rate, cache_efficiency):
    #     total_requests = len(request_dataset)
    #     num_vehicles = self.args.vehicle_num
    #     total_v2i_rate = sum(v2i_rate[:num_vehicles])
    #     avg_v2i_rate = total_v2i_rate / num_vehicles if num_vehicles > 0 else 1  # Avoid division by zero
    #
    #     delay_cached = (total_requests / avg_v2i_rate) * 800  # Delay for cached content
    #     delay_uncached = (total_requests / (avg_v2i_rate / 2)) * 800  # Delay for uncached content
    #     average_delay = cache_efficiency * delay_cached + (1 - cache_efficiency) * delay_uncached
    #     average_delay_ms = average_delay * 1000  # Convert to milliseconds
    #
    #     # Introduce weights for cache efficiency and delay
    #     alpha = 1.0  # weight for cache efficiency
    #     beta = 0.01  # weight for delay (adjust as needed)
    #
    #     reward = alpha * cache_efficiency - beta * average_delay_ms
    #
    #     return reward, average_delay_ms

    def calculate_reward_and_delay(self, request_dataset, v2i_rate, cache_efficiency):
        reward = 0
        request_delay = 0
        hit_ratio_weight = 2 / 3  # Weight for hit ratio contribution to the reward
        delay_weight = 1 / 3  # Weight for delay contribution to the reward

        for i in range(self.args.vehicle_num):
            vehicle_idx = i
            hit_ratio_reward = cache_efficiency * hit_ratio_weight
            delay_reward = (1 - cache_efficiency) * delay_weight * math.exp(-0.0001 * 8000000 / v2i_rate[vehicle_idx])

            reward += (hit_ratio_reward + delay_reward) * len(request_dataset)

            # Delay calculations adjusted to only account for cache_efficiency
            request_delay += cache_efficiency * len(request_dataset) / v2i_rate[vehicle_idx] * 800
            request_delay += (1 - cache_efficiency) * (
                    len(request_dataset) / (v2i_rate[vehicle_idx] / 2)) * 800

        request_delay = request_delay / self.args.vehicle_num * 1000
        return reward, request_delay


    def reset(self):

        return (self.init_state,
                self.init_edge_index,
                )
