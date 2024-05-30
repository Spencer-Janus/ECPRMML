# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 01:10:01 2023

@author: Janus_yu
"""

from .metrics import *
from .parser import parse_args

import torch
import numpy as np
import multiprocessing
import heapq
from time import time
import pandas as pd
import torch.nn.functional as F
args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks,u):
    item_score = {}
    u=u%BATCH_SIZE
    for i in test_items:

        # item_score[i] = list(rating[i])
        item_score[i] = rating[u,i]

        # item_score[i] = list(rating[u,i])

    K_max = max(Ks)

    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    # print(K_max_item_score)
    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    # print(r)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)

    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        # ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x,y):
    # user u's ratings for user u
    rating = x
    # uid
    u =y
    # print(u)
    # user u's items in the training set
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []

    # user u's items in the test set
    # user_pos_test = test_user_set[u]
    user_pos_test = test_user_set[u]
    # print(user_pos_test)
    all_items = set(range(0, n_items))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks,u)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks,u)

    return get_performance(user_pos_test, r, auc, Ks)
def softmax(X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(1, keepdim=True)
        return X_exp / partition
def test(model, user_dict, n_params,agg):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    entity_gcn_emb, item_emb_last,w = model.generate()
    num=0
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_list_batch = test_users[start: end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = entity_gcn_emb[user_batch]
        # print(u_g_embeddings.shape)
        if batch_test_flag:
            # batch-item test
            n_item_batchs = n_items // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_items))
            # print(rate_batch.shape)
            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(device)
                i_g_embddings = item_emb_last[item_batch]
                # i_g_embddings = item_emb_last(item_batch)

                i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == n_items
        else:
            # all-item test
            item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
            i_g_embddings = item_emb_last(item_batch)
            rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()
        # print(user_list_batch)
        # print(rate_batch.shape)
        batch_result=[]
        for i in user_list_batch:
            batch_result.append(test_one_user(rate_batch, i))
            num=num+1
        count += len(batch_result)
        # print(count)

        for re in batch_result:
            result['precision'] += re['precision']
            result['recall'] += re['recall']
            # result['ndcg'] += re['ndcg']
            result['hit_ratio'] += re['hit_ratio']
            result['auc'] += re['auc']

    result['precision'] = result['precision']/n_test_users
    result['recall'] = result['recall']/n_test_users
            # result['ndcg'] += re['ndcg']
    result['hit_ratio'] += result['hit_ratio']/n_test_users
    result['auc'] += result['auc']/result['hit_ratio']
    # print(result)
    return result
