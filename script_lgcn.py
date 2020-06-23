import time

import dgl
import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np

from lgcn_dataset import DataOnlyCF
from lgcn_model import LightGCN
from metrics import precision_and_recall, ndcg, auc

EPOCH = 200
LR = 0.001
EDIM = 64
LAYERS = 3
LAM = 1e-4
TOPK = 20

# GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, data_loader, optimizer, log_interval=10):
    model.train()
    total_loss = 0
    for i, (user_ids, pos_ids, neg_ids) in enumerate(tqdm.tqdm(data_loader)):
    # for i, (user_ids, pos_ids, neg_ids) in enumerate(data_loader):
        user_ids = user_ids.to(device)
        pos_ids = pos_ids.to(device)
        neg_ids = neg_ids.to(device)
        loss = model.bpr_loss(user_ids, pos_ids, neg_ids)
        # print('train loss ' + str(i) + '/' + str(len(data_loader)) + ': ' + str(loss))
        model.zero_grad()
        time_start = time.time()
        loss.backward()
        # print('loss.backward time:' + str(time.time() - time_start))
        optimizer.step()
        total_loss += loss.cpu().item()
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
            user_ids = user_ids.to(device)
            pos_ids = pos_ids.to(device)
            neg_ids = neg_ids.to(device)
            loss = model.bpr_loss(user_ids, pos_ids, neg_ids)
            total_loss += loss.cpu().item()
        avg_loss = total_loss / len(data_loader)
        print('evaluate loss:' + str(avg_loss))

def test(data_set, model, data_loader):
    with torch.no_grad():
        print('----- start_test -----')
        model.eval()
        precision = []
        recall = []
        ndcg_score = []
        auc_score = []
        for user_ids, _, __ in tqdm.tqdm(data_loader):
            user_ids = user_ids.to(device)
            ratings = model.get_users_ratings(user_ids)
            ground_truths = []
            for i, user_id_t in enumerate(user_ids):
                user_id = user_id_t.item()
                ground_truths.append(data_set.test_user_dict[user_id])
                train_pos = data_set.train_user_dict[user_id]
                for pos_item in train_pos:
                    ratings[i][pos_item] = -1 # delete train data in ratings
            # Precision, Recall, NDCG
            ___, index_k = torch.topk(ratings, k=TOPK) # index_k.shape = (batch_size, TOPK), dtype=torch.int
            batch_predict_items = index_k.cpu().tolist()
            batch_precision, batch_recall = precision_and_recall(batch_predict_items, ground_truths)
            batch_ndcg = ndcg(batch_predict_items, ground_truths)
            # AUC
            ratings = ratings.cpu().numpy()
            batch_auc = auc(ratings, data_set.get_item_num(), ground_truths)

            precision.append(batch_precision)
            recall.append(batch_recall)
            ndcg_score.append(batch_ndcg)
            auc_score.append(batch_auc)
        precision = np.mean(precision)
        recall = np.mean(recall)
        ndcg_score = np.mean(ndcg_score)
        auc_score = np.mean(auc_score)
        print('test result: precision ' + str(precision) + '; recall ' + str(recall) + '; ndcg ' + str(ndcg_score) + '; auc ' + str(auc_score))


if __name__ == "__main__":
    data_set = DataOnlyCF('data_lgcn/gowalla/train.txt', 'data_lgcn/gowalla/test.txt')
    G = data_set.get_interaction_graph()
    G.ndata['id'] = G.ndata['id'].to(device) # move graph data to target device
    G.ndata['sqrt_degree'] = G.ndata['sqrt_degree'].to(device) # move graph data to target device
    n_users = data_set.get_user_num()
    n_items = data_set.get_item_num()
    model = LightGCN(n_users, n_items, G, embed_dim=EDIM, n_layers=LAYERS, lam=LAM).to(device)
    train_data_loader = DataLoader(data_set, batch_size=2048, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(data_set.get_test_dataset(), batch_size=4096, num_workers=4)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    test(data_set, model, test_data_loader)######
    exit(0)######
    for epoch_i in range(EPOCH):
        print('Train lgcn - epoch ' + str(epoch_i + 1) + '/' + str(EPOCH))
        train(model, train_data_loader, optimizer)
        evaluate(model, test_data_loader)
        if (epoch_i + 1) % 10 == 0:
            test(data_set, model, test_data_loader)
        print('--------------------------------------------------')
    print('==================================================')
    test(data_set, model, test_data_loader)

# run data/amazon
# train loss 0.11; evaluate loss 0.23
# precision 0.0084; recall 0.0781

# run data_lgcn/gowalla
# train loss 0.07; evaluate loss 0.11
# precision 0.026320556846477126; recall 0.0931933408059992

# Paper code at epoch 50 gowalla
# {'precision': array([0.04382075]), 'recall': array([0.14503336]), 'ndcg': array([0.12077126]), 'auc': 0.9587075653077938}
# Paper code at epoch 80 gowalla
# {'precision': array([0.0468166]), 'recall': array([0.15585551]), 'ndcg': array([0.13010746]), 'auc': 0.9582199598920466}
