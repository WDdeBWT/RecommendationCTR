import time

import dgl
import torch
from torch.utils.data import DataLoader
import numpy as np

class TestDatasetOnlyCF(torch.utils.data.Dataset):

    def __init__(self, train_user_dict, test_user_dict, test_user_list, n_items):
        self.train_user_dict = train_user_dict
        self.test_user_dict = test_user_dict
        self.test_user_list = test_user_list
        self.n_items = n_items

    def __len__(self):
        return len(self.test_user_list)

    def __getitem__(self, index):
        # Problem: not traversal, but sample
        user_id = self.test_user_list[index]
        pos_id = self.test_user_dict[user_id][np.random.randint(0, len(self.test_user_dict[user_id]))]
        while True:
            neg_id = np.random.randint(0, self.n_items)
            if neg_id in self.train_user_dict[user_id]:
                continue
            elif neg_id in self.test_user_dict[user_id]:
                continue
            else:
                break
        return user_id, pos_id, neg_id

class DataOnlyCF(torch.utils.data.Dataset):

    def __init__(self, train_data_path, test_data_path):
        self.train_data, self.train_user_dict = self._load_cf_data(train_data_path)
        self.test_data, self.test_user_dict = self._load_cf_data(test_data_path)
        self.train_user_list = list(self.train_user_dict.keys())
        self.test_user_list = list(self.test_user_dict.keys())
        self.n_users, self.n_items, self.n_train, self.n_test = self._statistic_cf()
        self.G = self._build_interaction_graph()

    def _load_cf_data(self, file_path):
        cases_user = []
        cases_item = []
        user_dict = dict()

        lines = open(file_path, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))

                for item_id in item_ids:
                    cases_user.append(user_id)
                    cases_item.append(item_id)
                user_dict[user_id] = item_ids # {user_id: [item_ids]}

        cases_user = np.array(cases_user, dtype=np.int32)
        cases_item = np.array(cases_item, dtype=np.int32)
        return [cases_user, cases_item], user_dict

    def _statistic_cf(self):
        n_users = max(max(self.train_data[0]), max(self.test_data[0])) + 1
        n_items = max(max(self.train_data[1]), max(self.test_data[1])) + 1
        n_train = len(self.train_data[0])
        n_test = len(self.test_data[0])
        return n_users, n_items, n_train, n_test

    def _build_interaction_graph(self):
        n_nodes = self.n_users + self.n_items
        g = dgl.DGLGraph()
        g.add_nodes(n_nodes)
        # item id start from self.n_users
        g.add_edges(self.train_data[0], self.train_data[1] + self.n_users)
        g.add_edges(self.train_data[1] + self.n_users, self.train_data[0])
        g.readonly()
        g.ndata['id'] = torch.arange(n_nodes, dtype=torch.long)
        g.ndata['sqrt_degree'] = 1 / torch.sqrt(g.out_degrees().float().unsqueeze(-1))
        return g

    # def __len__(self): # first version
    #     return len(self.train_user_list)

    # def __getitem__(self, index): # first version
    #     # Problem: not traversal, but sample
    #     user_id = self.train_user_list[index]
    #     pos_id = self.train_user_dict[user_id][np.random.randint(0, len(self.train_user_dict[user_id]))]
    #     while True:
    #         neg_id = np.random.randint(0, self.n_items)
    #         if neg_id in self.train_user_dict[user_id]:
    #             continue
    #         else:
    #             break
    #     return user_id, pos_id, neg_id

    def __len__(self): # secend/third version
        return self.n_train

    def __getitem__(self, index): # secend version
        user_id = np.random.randint(0, self.n_users)
        pos_id = self.train_user_dict[user_id][np.random.randint(0, len(self.train_user_dict[user_id]))]
        while True:
            neg_id = np.random.randint(0, self.n_items)
            if neg_id in self.train_user_dict[user_id]:
                continue
            else:
                break
        return user_id, pos_id, neg_id

    # def __getitem__(self, index): # third version
    #     user_id = self.train_data[0][index]
    #     pos_id = self.train_data[1][index]
    #     while True:
    #         neg_id = np.random.randint(0, self.n_items)
    #         if neg_id in self.train_user_dict[user_id]:
    #             continue
    #         else:
    #             break
    #     return user_id, pos_id, neg_id

    def get_interaction_graph(self):
        return self.G

    def get_user_num(self):
        return self.n_users

    def get_item_num(self):
        return self.n_items

    def get_train_data(self):
        return self.train_data

    def get_test_dataset(self):
        return TestDatasetOnlyCF(self.train_user_dict, self.test_user_dict, self.test_user_list, self.n_items)


if __name__ == "__main__":
    data = DataOnlyCF('data/amazon-book/train.txt', 'data/amazon-book/test.txt')
    G = data.G
    print(G)
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    data_loader = DataLoader(data, batch_size=8, num_workers=2)
    for u, p, n in data_loader:
        print(u)
        print()
        print(p)
        print()
        print(n)
        print('------------------------------')
        time.sleep(5)