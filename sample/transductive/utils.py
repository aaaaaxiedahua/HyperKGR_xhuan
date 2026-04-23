import numpy as np
from scipy.stats import rankdata
import os

def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

def cal_ranks(scores, labels, filters):
    scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)
    scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-8
    full_rank = rankdata(-scores, method='average', axis=1)
    filter_scores = scores * filters
    filter_rank = rankdata(-filter_scores, method='min', axis=1)
    ranks = (full_rank - filter_rank + 1) * labels
    ranks = ranks[np.nonzero(ranks)]
    return list(ranks)

def cal_performance(ranks):
    if len(ranks) == 0:
        return 0.0, 0.0, 0.0
    ranks = np.asarray(ranks)
    ranks = ranks[np.isfinite(ranks)]
    if len(ranks) == 0:
        return 0.0, 0.0, 0.0
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks<=1) * 1.0 / len(ranks)
    h_10 = sum(ranks<=10) * 1.0 / len(ranks)
    return mrr, h_1, h_10

def uniqueWithoutSort(a):
    indexes = np.unique(a, return_index=True)[1]
    res = [a[index] for index in sorted(indexes)]
    return res
