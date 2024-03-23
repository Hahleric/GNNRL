import numpy

import environment
import GCNAgent as agent
from model import ActorGCN, CriticGCN
import numpy as np
import torch


# generate some pseudo data and test environment, model, agents

if __name__ == '__main__':
    popular_file = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cache_size = 3
    recommend_list = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                [1, 2, 3, 4, 5, 6, 7, 18, 9, 10],
                                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    env = environment.Environment(cache_size, popular_file, recommend_list)
    env.reset()
    env.step(1, numpy.array([[4,4,5], [4,4,5], [4,4,5]]), [1,1,1], [1,1,1], [1,1,1], [1,1,1], 50)


