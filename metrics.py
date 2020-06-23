import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def precision_and_recall(batch_predict_items, batch_truth_items):
    assert len(batch_predict_items) == len(batch_truth_items)
    precision = []
    recall = []
    for predict_items, truth_items in zip(batch_predict_items, batch_truth_items):
        hit = 0
        for predict_item in predict_items:
            if predict_item in truth_items:
                hit += 1
        precision.append(hit / len(predict_items))
        recall.append(hit / len(truth_items))
    return np.mean(precision).item(), np.mean(recall).item()

def ndcg(batch_predict_items, batch_truth_items):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(batch_predict_items) == len(batch_truth_items) # batch_size
    batch_predict_label = []
    for i, predict_items in enumerate(batch_predict_items):
        predict_label = [item in batch_truth_items[i] for item in predict_items]
        batch_predict_label.append(predict_label)
    batch_predict_label = np.array(batch_predict_label, dtype=int)
    k = batch_predict_label.shape[1] # top_k

    batch_ideal_label = np.zeros((len(batch_truth_items), k))
    for i, items in enumerate(batch_truth_items):
        length = min(k, len(items))
        batch_ideal_label[i, :length] = 1
    idcg = np.sum(batch_ideal_label * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = batch_predict_label*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.mean(ndcg)

def auc(ratings, n_items, batch_truth_items):
    """
        design for a single user
    """
    auc_scores = []
    for i, truth_items in enumerate(batch_truth_items):
        all_item_scores = ratings[i]
        r_all = np.zeros((n_items, ))
        r_all[truth_items] = 1
        r = r_all[all_item_scores >= 0]
        test_item_scores = all_item_scores[all_item_scores >= 0]
        auc_scores.append(roc_auc_score(r, test_item_scores))
    return np.mean(auc_scores).item()
