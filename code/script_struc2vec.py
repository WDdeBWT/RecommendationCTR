import numpy as np

from s2vec.struc2vec import Struc2Vec
from cf_dataset import DataOnlyCF

import networkx as nx


def get_rec_graph(train_path, test_path):
    data_set = DataOnlyCF(train_path, test_path)
    n_users = data_set.get_user_num()
    n_items = data_set.get_item_num()
    train_data = data_set.get_train_data()
    edges = np.concatenate((train_data[0].reshape(-1, 1), train_data[1].reshape(-1, 1) + n_users), 1)
    G = nx.Graph()
    G.add_nodes_from(range(n_users + n_items))
    G.add_edges_from(edges)
    print('Nx graph build finish, n_nodes: ' + str(len(G.nodes)) + '; n_edges: ' + str(len(G.edges)))
    return G


if __name__ == "__main__":
    rec_graph = get_rec_graph('data_lgcn/gowalla/train.txt', 'data_lgcn/gowalla/test.txt')

    s2v = Struc2Vec(rec_graph, workers=4, verbose=40, opt3_num_layers=3, reuse=True)
    g_list = s2v.get_struc_graphs()
    for i, g in enumerate(g_list):
        print(i, len(g.nodes), len(g.edges))
