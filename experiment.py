import numpy

import GCNAgent
import environment
import GCNAgent as Agent
from GCNAgent import mini_batch_train
from model import ActorGCN, CriticGCN
from utils.data_preprocess import create_rating_matrix, get_ratings, create_top_100_rating_matrix, get_user_rated_movies
import numpy as np
import torch

if __name__ == '__main__':
    # c-v2x simulation parameters:
    V2I_min = 100  # minimum required data rate for V2I Communication
    bandwidth = int(540000)
    bandwidth_mbs = int(1000000)

    data_set_path = 'ml-10M100K/ratings.dat'
    user_ratings = create_top_100_rating_matrix(data_set_path)
    cache_size = 40
    request_data = get_user_rated_movies(data_set_path)
    # TODO currently, randomly select some users to train the model
    sampled_users = np.random.choice(user_ratings.shape[0], 100)
    env = environment.Environment(40, sampled_users)
    agent = GCNAgent.GCNAgent(cache_size, 32)
    reward, cache_efficiency, request_delay = mini_batch_train(env, agent, 30, 100, 32, request_data, 0, 0, 0, 0)




