import random

import numpy as np
from utils.cache_utils import cache_hit_ratio, cache_hit_ratio2
import random
import math


class Environment():
    def __init__(self, cache_size, popular_file, recommend_list):
        """
        :param cache_size: cache size of the RSU, which is the only agent in our environment
        :param popular_file: the sorted movie list by popularity, generated by recommender as well, it is
                            the sum of all movie rates, and sorted by the rate. Movies! not rates!
        :param recommend_list: popular contents generated by recommender system. It is a sorted list of
                            user preferred movies
        """
        self.cache_size = cache_size
        self.popular_file = popular_file
        self.action_space = [0, 1]
        self.reward = 0
        self.state = []

        if len(self.popular_file) < self.cache_size:
            self.state += self.popular_file

        if len(self.popular_file) > self.cache_size:
            self.state += random.sample(list(self.popular_file), self.cache_size)

        state = []
        for i in range(len(self.popular_file)):
            # 按照内容流行度进行排序
            if self.popular_file[i] in self.state:
                state.append(self.popular_file[i])
        self.state = state

        remaining_content = []
        for i in range(len(self.popular_file)):
            if self.popular_file[i] not in self.state:
                remaining_content.append(self.popular_file[i])
        self.remaining_content = remaining_content
        print('self.cache_size', self.cache_size)
        print('remaining contents', self.remaining_content)
        # recommend_list size (n, m), add first cache_size elements of every n row to state
        self.init_state = self.state.copy() + recommend_list[:, :self.cache_size]
        self.init_remaining_content = self.remaining_content.copy()

    def step(self, action, request_dataset, v2i_rate, v2i_rate_mbs, vehicle_epoch, vehicle_request_num, print_step):
        """
        :param action: action taken by the agent, generated by model
        :param request_dataset: user will request some contents from the RSU
        :param v2i_rate: vehicle to infrastructure rate
        :param v2i_rate_mbs: vehicle to infrastructure rate in MBS
        :param vehicle_epoch: current epoch for some vehicle
        :return: state_, reward, cache_efficiency, cache_efficiency2, request_delay
        """

        if action == 1:

            if len(self.remaining_content) >= 5:
                replace_content = random.sample(list(self.remaining_content), 5)
                count = 0
                if count < 5:
                    self.state[-count - 1] = replace_content[count]
                    count += 1
            else:
                replace_content = self.remaining_content
            count = 0
            if count < 5:
                self.state[-count - 1] = replace_content[count]
                count += 1

            state = []
            for i in range(len(self.popular_file)):
                # 按照内容流行度进行排序
                if self.popular_file[i] in self.state:
                    state.append(self.popular_file[i])
            self.state = state

            last_content = []
            for i in range(len(self.popular_file)):
                if self.popular_file[i] not in self.state:
                    last_content.append(self.popular_file[i])
            self.remaining_content = last_content

            all_vehicle_request_num = 0
            for i in range(len(vehicle_epoch)):
                all_vehicle_request_num += vehicle_request_num[vehicle_epoch[i]]
            # print('=================================all_vehicle_request_num', all_vehicle_request_num,
            # '================================')
            cache_efficiency = cache_hit_ratio(request_dataset, self.state,
                                               all_vehicle_request_num)
            cache_efficiency = cache_efficiency / 100

            reward = 0
            request_delay = 0
            for i in range(len(vehicle_epoch)):
                vehicle_idx = vehicle_epoch[i]

                # Only one cache efficiency calculation
                reward += cache_efficiency * math.exp(-0.0001 * 8000000 / v2i_rate[vehicle_idx]) * vehicle_request_num[
                    vehicle_idx]
                reward += (1 - cache_efficiency) * math.exp(-0.5999 * 8000000 / (v2i_rate[vehicle_idx] / 2)) * \
                          vehicle_request_num[vehicle_idx]

                # Delay calculations adjusted to only account for cache_efficiency
                request_delay += cache_efficiency * vehicle_request_num[vehicle_idx] / v2i_rate[vehicle_idx] * 800
                request_delay += (1 - cache_efficiency) * (
                        vehicle_request_num[vehicle_idx] / (v2i_rate[vehicle_idx] / 2)) * 800

                # print(i,'mbs delay',(vehicle_request_num[vehicle_idx] / v2i_rate_mbs[vehicle_idx]) *100000)
            request_delay = request_delay / len(vehicle_epoch) * 1000
            if print_step % 50 == 0:
                print("---------------------------------------------")
                print('all_vehicle_request_num', all_vehicle_request_num)
                print('step:{} RSU1 cache_efficiency:{}'.format(print_step, cache_efficiency))
                print('step', print_step, 'request delay:%f' % (request_delay))
                print("---------------------------------------------")
            return self.state, reward, cache_efficiency, request_delay

    def reset(self):
        return self.init_state, self.init_state, self.init_remaining_content
