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
    COVERED_VEH = 20
    REQUESTED_MOVIES = 20
    BATCH_SIZE = 32

    data_set_path = 'ml-latest-small/ratings.csv'
    user_ratings, top_num_rating_ndarray, request_data, top_num_popular_movies \
        = create_rating_matrix(data_set_path, REQUESTED_MOVIES)
    n_veh = user_ratings.shape[0]
    env_v = Environ(n_veh, V2I_min, bandwidth, bandwidth_mbs)
    vehicle_dis = np.random.normal(100, 50, n_veh)
    v2i_rate, v2i_rate_mbs = env_v.Compute_Performance_Static(vehicle_dis)


    cache_size = 20

    # TODO currently, randomly select some users to train the model
    # print request_data in a nice way
    random_idx = np.random.choice(n_veh, COVERED_VEH, replace=True)
    # sampled_requested_movies = np.array(request_data, shape=())[random_idx, :]
    sampled_v2i_rate = v2i_rate[random_idx]
    sampled_v2i_rate_mbs = v2i_rate_mbs[random_idx]
    sampled_request_movies = top_num_rating_ndarray[random_idx, :]
    sampled_veh_dis = vehicle_dis[random_idx]
    # node_features
    sampled_request_movies = np.nan_to_num(sampled_request_movies)
    env = environment.Environment(cache_size, top_num_popular_movies, sampled_request_movies)
    agent = Agent.GCNAgent(cache_size, 32)
    episode_rewards, cache_efficiency, request_delay = mini_batch_train(env, agent, 10, 10,
                                                                        32,
                                                                        sampled_request_movies,
                                                                        sampled_v2i_rate,
                                                                        sampled_veh_dis)









