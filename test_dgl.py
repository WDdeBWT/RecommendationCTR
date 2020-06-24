import time

import dgl
import dgl.function as fn
import torch
import networkx as nx
import matplotlib.pyplot as plt

def pagerank_message_func(edges):
    return {'pv' : edges.src['pv'] / edges.src['deg']}

def pagerank_reduce_func(nodes):
    msgs = torch.sum(nodes.mailbox['pv'], dim=1)
    pv = (1 - DAMP) / N + DAMP * msgs
    return {'pv' : pv}

def pagerank_naive(g):
    # Phase #1: send out messages along all edges.
    for u, v in zip(*g.edges()):
        g.send((u, v))
    # Phase #2: receive messages to compute new PageRank values.
    for v in g.nodes():
        g.recv(v)

def pagerank_batch(g):
    g.send(g.edges())
    g.recv(g.nodes())

def pagerank_builtin(g):
    g.ndata['pv'] = g.ndata['pv'] / g.ndata['deg']
    g.update_all(message_func=fn.copy_src(src='pv', out='m'),
                 reduce_func=fn.sum(msg='m',out='m_sum'))
    g.ndata['pv'] = (1 - DAMP) / N + DAMP * g.ndata['m_sum']

if __name__ == "__main__":
    N = 10000  # number of nodes
    DAMP = 0.85  # damping factor
    K = 10  # number of iterations
    g = nx.nx.erdos_renyi_graph(N, 0.1)
    g = dgl.DGLGraph(g)
    # nx.draw(g.to_networkx(), node_size=50, node_color=[[.5, .5, .5,]])
    # plt.show()

    g.ndata['pv'] = torch.ones(N) / N
    g.ndata['deg'] = g.out_degrees(g.nodes()).float()
    g.register_message_func(pagerank_message_func)
    g.register_reduce_func(pagerank_reduce_func)

    ts = time.time()
    for k in range(K):
        # pagerank_builtin(g)
        pagerank_naive(g)
    print('time spend: ' + str(time.time() - ts))

    # print(g.ndata['pv'])
