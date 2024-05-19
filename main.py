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

import GCNAgent
import experiment
from utils.parser import parse_args
from utils.dataloader import Dataloader
from utils.utils import config, construct_negative_graph, choose_model, load_mf_model, NegativeGraph
from utils.tester import Tester
from models.sampler import NegativeSampler
import environment

if __name__ == '__main__':
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
    opt = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    early_stop(99999.99, model)
    item = []
    graph = None
    score_pos = None
    for epoch in range(args.epoch):
        model.train()

        loss_train = torch.zeros(1).to(device)

        graph_pos = dataloader.train_graph
        for i in range(args.neg_number):
            graph_neg = construct_negative_graph(graph_pos, ('user', 'rate', 'item'))

            graph, score_pos, score_neg = model(graph_pos, graph_neg)
            if not args.category_balance:
                loss_train += -(score_pos - score_neg).sigmoid().log().mean()
            else:
                loss = -(score_pos - score_neg).sigmoid().log()
                items = graph_pos.edges(etype = 'rate')[1]
                weight = sample_weight[items]
                loss_train += (weight * loss.squeeze(1)).mean()

        # 每个正样本，采样args.neg_number个负样本

        loss_train = loss_train / args.neg_number
        # 平均一下
        logging.info('train loss = {}'.format(loss_train.item()))
        opt.zero_grad()
        loss_train.backward()
        opt.step()

        model.eval()
        graph_val_pos = dataloader.val_graph
        graph_val_neg = construct_negative_graph(graph_val_pos, ('user', 'rate', 'item'))

        graph, score_pos, score_neg = model(graph_val_pos, graph_val_neg)
        if not args.category_balance:
            loss_val = -(score_pos - score_neg).sigmoid().log().mean()
        else:
            loss = -(score_pos - score_neg).sigmoid().log()
            items = graph_val_pos.edges(etype = 'rate')[1]
            weight = sample_weight[items]
            loss_val = (weight * loss.squeeze(1)).mean()


        early_stop(loss_val, model)

        if torch.isnan(loss_val) == True:
            break

        if early_stop.early_stop:
            break

    logging.info('loading best model for RL')
    model.load_state_dict(torch.load(early_stop.save_path))
    top_scores, top_indices = torch.topk(score_pos.T, k=args.k_list, largest=True)
    src, dst = graph.edges(etype='rate')
    top_items = dst[top_indices]
    environment = environment.Environment(args, cache_size=args.cache_size, popular_file=top_items)
    h = model.get_embedding()
    agent = GCNAgent.GCNAgent(args)
    exp = experiment.Experiment(args, model, dataloader, environment, agent)
    episode_rewards, cache_efficiency, request_delay = exp.start()

