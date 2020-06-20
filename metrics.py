import torch
from numpy import mean

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
    return mean(precision).item(), mean(recall).item()