import time

import dgl
import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np

from lgcn_dataset import DataOnlyCF
from lgcn_model import LightGCN
from metrics import precision_and_recall

EPOCH = 200
LR = 0.001
EDIM = 64
LAYERS = 3
LAM = 0.001
TOPK = 20

def train(model, data_loader, optimizer, log_interval=10):
    model.train()
    total_loss = 0
    for i, (user_ids, pos_ids, neg_ids) in enumerate(tqdm.tqdm(data_loader)):
    # for i, (user_ids, pos_ids, neg_ids) in enumerate(data_loader):
        loss = model.bpr_loss(user_ids, pos_ids, neg_ids)
        # print('train loss ' + str(i) + '/' + str(len(data_loader)) + ': ' + str(loss))
        model.zero_grad()
        time_start = time.time()
        loss.backward()
        # print('loss.backward time:' + str(time.time() - time_start))
        optimizer.step()
        total_loss += loss.item()
        # if (i + 1) % log_interval == 0:
        #     print('    - Average loss:', total_loss / log_interval)
        #     total_loss = 0
    print('train loss:', total_loss / len(data_loader))

def evaluate(model, data_loader):
    with torch.no_grad():
        # print('----- start_evaluate -----')
        model.eval()
        total_loss = 0
        # for i, (user_ids, pos_ids, neg_ids) in enumerate(tqdm.tqdm(data_loader)):
        for i, (user_ids, pos_ids, neg_ids) in enumerate(data_loader):
            loss = model.bpr_loss(user_ids, pos_ids, neg_ids)
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print('evaluate loss:' + str(avg_loss))

def test(data_set, model, data_loader):
    with torch.no_grad():
        print('----- start_test -----')
        model.eval()
        precision = []
        recall = []
        for i, (user_ids, _, __) in enumerate(tqdm.tqdm(data_loader)):
            ratings = model.get_users_ratings(user_ids)
            ground_truth = []
            for user_id_t in user_ids:
                user_id = user_id_t.item()
                ground_truth.append(data_set.test_user_dict[user_id])
                train_pos = data_set.train_user_dict[user_id]
                for pos_item in train_pos:
                    ratings[i][pos_item] = 0 # delete train data in ratings
            ___, index_k = torch.topk(ratings, k=TOPK) # index_k.shape = (batch_size, TOPK), dtype=torch.int
            batch_precision, batch_recall = precision_and_recall(index_k.tolist(), ground_truth)
            precision.append(batch_precision)
            recall.append(batch_recall)
        precision = np.mean(precision)
        recall = np.mean(recall)
        print('test result: precision ' + str(precision) + '; recall ' + str(recall))


if __name__ == "__main__":
    data_set = DataOnlyCF('data/amazon-book/train.txt', 'data/amazon-book/test.txt')
    G = data_set.get_interaction_graph()
    n_users = data_set.get_user_num()
    n_items = data_set.get_item_num()
    model = LightGCN(n_users, n_items, G, embed_dim=EDIM, n_layers=LAYERS, lam=LAM)
    train_data_loader = DataLoader(data_set, batch_size=2048, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(data_set.get_test_dataset(), batch_size=4096, num_workers=4)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    for epoch_i in range(EPOCH):
        print('Train lgcn - epoch ' + str(epoch_i) + '/' + str(EPOCH))
        train(model, train_data_loader, optimizer)
        evaluate(model, test_data_loader)
        print('--------------------------------------------------')
    print('==================================================')
    test(data_set, model, test_data_loader)

# train loss 0.11; evaluate loss 0.23
# precision 0.0084; recall 0.0781