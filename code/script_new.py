import time

import dgl
import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np

from cf_dataset import DataOnlyCF
from gcn_model import CFGCN
from metrics import precision_and_recall, ndcg, auc

EPOCH = 500
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

def test(data_set, model, data_loader, show_auc = False):
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
            if show_auc:
                ratings = ratings.cpu().numpy()
                batch_auc = auc(ratings, data_set.get_item_num(), ground_truths)
                auc_score.append(batch_auc)

            precision.append(batch_precision)
            recall.append(batch_recall)
            ndcg_score.append(batch_ndcg)
        precision = np.mean(precision)
        recall = np.mean(recall)
        ndcg_score = np.mean(ndcg_score)
        if show_auc: # Calculate AUC scores spends a long time
            auc_score = np.mean(auc_score)
            print('test result: precision ' + str(precision) + '; recall ' + str(recall) + '; ndcg ' + str(ndcg_score) + '; auc ' + str(auc_score))
        else:
            print('test result: precision ' + str(precision) + '; recall ' + str(recall) + '; ndcg ' + str(ndcg_score))


if __name__ == "__main__":
    data_set = DataOnlyCF('data_for_test/gowalla/train.txt', 'data_for_test/gowalla/test.txt')
    itra_G = data_set.get_interaction_graph()
    itra_G.ndata['id'] = itra_G.ndata['id'].to(device) # move graph data to target device
    itra_G.ndata['sqrt_degree'] = itra_G.ndata['sqrt_degree'].to(device) # move graph data to target device
    struc_Gs = data_set.build_struc_graphs(sumed=True)
    # for g in struc_Gs:
    #     g.ndata['id'] = g.ndata['id'].to(device)
    #     g.edata['weight'] = g.edata['weight'].to(device)
    struc_Gs = None
    n_users = data_set.get_user_num()
    n_items = data_set.get_item_num()
    model = CFGCN(n_users, n_items, itra_G, struc_Gs=struc_Gs, embed_dim=EDIM, n_layers=LAYERS, lam=LAM).to(device)
    train_data_loader = DataLoader(data_set, batch_size=2048, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(data_set.get_test_dataset(), batch_size=4096, num_workers=4)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    for epoch_i in range(EPOCH):
        print('Train lgcn - epoch ' + str(epoch_i + 1) + '/' + str(EPOCH))
        train(model, train_data_loader, optimizer)
        evaluate(model, test_data_loader)
        if (epoch_i + 1) % 10 == 0:
            test(data_set, model, test_data_loader)
        print('--------------------------------------------------')
    print('==================================================')
    test(data_set, model, test_data_loader)

# run data_lgcn/gowalla gowalla
# at epoch 50 precision 0.0406273132632997; recall 0.13624640704870125; ndcg 0.11335605664660738
# at epoch 100 precision 0.043916843150558604; recall 0.1494013824889015; ndcg 0.12420528799591576
# at epoch 200 precision 0.04701247577120442; recall 0.16131859933809437; ndcg 0.13461157598002454
# at epoch 300 precision 0.04827474044105054; recall 0.1658744830844509; ndcg 0.13881823365718912
# at epoch 400 precision 0.04889910796038417; recall 0.16776382478552415; ndcg 0.14091457113502115
# at epoch 500 precision 0.04972081819634011; recall 0.17046915726601; ndcg 0.14296358168788104
# epoch 300 (old)
# train loss 0.015; evaluate loss 0.134
# test result: precision 0.047575934218717066; recall 0.16351703048292573; ndcg 0.13673274095554458
# max
# recall 0.172; ndcg 0.143

# Paper code at epoch 50 gowalla
# {'precision': array([0.04382075]), 'recall': array([0.14503336]), 'ndcg': array([0.12077126]), 'auc': 0.9587075653077938}
# Paper code at epoch 80 gowalla
# {'precision': array([0.0468166]), 'recall': array([0.15585551]), 'ndcg': array([0.13010746]), 'auc': 0.9582199598920466}
# Paper code at epoch 400 gowalla
# {'precision': array([0.05400898]), 'recall': array([0.17730673]), 'ndcg': array([0.15099276]), 'auc': 0.9508640011701547}
