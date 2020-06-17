import time

import dgl
import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np

from lgcn_dataset import DataOnlyCF
from lgcn_model import LightGCN

EPOCH = 10
LR = 0.001

def train(model, data_loader, optimizer, G, log_interval=10):
    model.train()
    total_loss = 0
    # for i, (user_ids, pos_ids, neg_ids) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
    for i, (user_ids, pos_ids, neg_ids) in enumerate(data_loader):
        loss = model.bpr_loss(user_ids, pos_ids, neg_ids, G)
        print(f'train loss {i + 1}/{len(data_loader)}: {loss}')
        model.zero_grad()
        time_start = time.time()
        loss.backward()
        print('loss.backward time:' + str(time.time() - time_start))
        optimizer.step()
        total_loss += loss.item()
        # if (i + 1) % log_interval == 0:
        #     print('    - loss:', total_loss / log_interval)
        #     total_loss = 0
        # print('train: ' + str(i) + 'finish')

if __name__ == "__main__":
    data = DataOnlyCF('data/amazon-book/train.txt', 'data/amazon-book/test.txt')
    G = data.get_interaction_graph()
    n_users = data.get_user_num()
    n_items = data.get_item_num()
    model = LightGCN(n_users, n_items, embed_dim=64, n_layers=3, lam=0.001)
    data_loader = DataLoader(data, batch_size=2048, num_workers=0)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6)
    for epoch_i in range(EPOCH):
        train(model, data_loader, optimizer, G)
    

