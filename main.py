import sys
import dgl
import dgl.function as fn
import os
import multiprocessing as mp
from tqdm import tqdm
import pdb
import numpy as np
import torch
import torch.nn as nn
import logging
import gc  # Python垃圾回收模块
from GCNAgent import GCNAgent, TransAgent, GATAgent, MLPAgent, DDQNAgent
import experiment
from utils.parser import parse_args
from utils.dataloader import Dataloader
from utils.utils import config, construct_negative_graph, choose_model, load_mf_model, NegativeGraph
from utils.tester import Tester
from models.sampler import NegativeSampler
import environment
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    early_stop = config(args)
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    device = torch.device(device)
    args.device = device

    data = args.dataset
    dataloader = Dataloader(args, data, device)
    # NegativeGraphConstructor = NegativeGraph(dataloader.historical_dict)
    sample_weight = dataloader.sample_weight.to(device)

    model = choose_model(args, dataloader)
    model = model.to(device)
    # opt = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    # early_stop(99999.99, model)
    # item = []
    # graph = None
    # score_pos = None
    # for epoch in range(args.epoch):
    #     model.train()
    #
    #     loss_train = torch.zeros(1).to(device)
    #
    #     graph_pos = dataloader.train_graph
    #     for i in range(args.neg_number):
    #         graph_neg = construct_negative_graph(graph_pos, ('user', 'rate', 'item'))
    #
    #         graph, score_pos, score_neg = model(graph_pos, graph_neg)
    #         if not args.category_balance:
    #             loss_train += -(score_pos - score_neg).sigmoid().log().mean()
    #         else:
    #             loss = -(score_pos - score_neg).sigmoid().log()
    #             items = graph_pos.edges(etype = 'rate')[1]
    #             weight = sample_weight[items]
    #             loss_train += (weight * loss.squeeze(1)).mean()
    #
    #     # 每个正样本，采样args.neg_number个负样本
    #
    #     loss_train = loss_train / args.neg_number
    #     # 平均一下
    #     logging.info('train loss = {}'.format(loss_train.item()))
    #     opt.zero_grad()
    #     loss_train.backward()
    #     opt.step()
    #
    #     model.eval()
    #     graph_val_pos = dataloader.val_graph
    #     graph_val_neg = construct_negative_graph(graph_val_pos, ('user', 'rate', 'item'))
    #
    #     graph, score_pos, score_neg = model(graph_val_pos, graph_val_neg)
    #     if not args.category_balance:
    #         loss_val = -(score_pos - score_neg).sigmoid().log().mean()
    #     else:
    #         loss = -(score_pos - score_neg).sigmoid().log()
    #         items = graph_val_pos.edges(etype = 'rate')[1]
    #         weight = sample_weight[items]
    #         loss_val = (weight * loss.squeeze(1)).mean()
    #
    #
    #     early_stop(loss_val, model)
    #
    #     if torch.isnan(loss_val) == True:
    #         break
    #
    #     if early_stop.early_stop:
    #         break

    logging.info('loading best model for RL')
    model.load_state_dict(torch.load('best_models/Beauty_model_dgrec_lr_0.05_embed_size_32_batch_size_20_weight_decay_8e-08_layers_1_neg_number_4_seed_2024_k_20_sigma_1.0_gamma_2.0_beta_class_0.9.pt'))
    test_items = dataloader.test_items
    environment = environment.Environment(args, cache_size=args.cache_size, test_items=test_items)
    h = model.get_embedding()
    # agent = GCNAgent(args)
    # agent_trans = TransAgent(args)
    # agent_gat = GATAgent(args)
    # MLPAgent = MLPAgent(args)
    # agents = [agent, agent_gat, agent_trans, MLPAgent]
    cache_efficiencies = []

    for i in range(1):
        if i == 0:
            agent = DDQNAgent(args)
        elif i == 1:
            agent = TransAgent(args)
        elif i == 2:
            agent = GCNAgent(args)
        else:
            agent = MLPAgent(args)
        exp = experiment.Experiment(args, model, dataloader, environment, agent)
        print('start regular with recommender')
        episode_rewards, cache_efficiency, request_delay, fifo_eff, lru_eff = exp.start_regular_with_recommender()
        cache_efficiencies.append(cache_efficiency)

        del agent
        del exp
        torch.cuda.empty_cache()  # 释放CUDA内存
        gc.collect()  # 强制执行垃圾回收
    plt.figure(figsize=(10, 5))

    plt.plot(range(len(cache_efficiencies[0])), cache_efficiencies[0], label='GCN')
    plt.plot(range(len(fifo_eff)), fifo_eff, label='FIFO')
    plt.plot(range(len(lru_eff)), lru_eff, label='LRU')
    plt.legend()
    plt.savefig('cache_efficiency.png')
    plt.xlabel('Step')
    plt.ylabel('Cache Efficiency')
    plt.title('Cache Efficiency per Step in the Last Episode')

    plt.grid(True)
    plt.show()




