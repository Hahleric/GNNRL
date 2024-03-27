import numpy
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import GCNAgent
import environment
import GCNAgent as Agent
from GCNAgent import mini_batch_train
from model import ActorGCN, CriticGCN
from utils.data_preprocess import create_rating_matrix
import numpy as np
import torch
from cv2x import V2Ichannels, Environ

if __name__ == '__main__':
    # c-v2x simulation parameters:

    V2I_min = 100  # minimum required data rate for V2I Communication
    bandwidth = int(540000)
    bandwidth_mbs = int(1000000)

    data_set_path = 'ml-latest-small/ratings.csv'
    user_ratings, top_100_rating_ndarray, request_data, top_100_popular_movies \
        = create_rating_matrix(data_set_path)
    print(user_ratings.shape)
    n_veh = user_ratings.shape[0]
    env_v = Environ(n_veh, V2I_min, bandwidth, bandwidth_mbs)
    vehicle_dis = np.zeros(n_veh)
    v2i_rate, v2i_rate_mbs = env_v.Compute_Performance_Static(vehicle_dis)
    cache_size = 40
    # TODO currently, randomly select some users to train the model
    sampled_users = np.random.choice(user_ratings.shape[0], 100)
    print(sampled_users.shape)
    env = environment.Environment(40, top_100_popular_movies, user_ratings)
    agent = GCNAgent.GCNAgent(cache_size, 32)
    reward, cache_efficiency, request_delay = mini_batch_train(env, agent, 30, 100, 32, request_data, v2i_rate, v2i_rate_mbs, [i for i in range(len(user_ratings))], request_data)




