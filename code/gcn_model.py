import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, aggregator_type='gcn'):
        super(Aggregator, self).__init__()
        self.aggregator_type = aggregator_type

        # self.dropout = dropout
        # self.message_dropout = nn.Dropout(dropout)

        if aggregator_type == 'gcn':
            self.W = nn.Linear(in_dim, out_dim)       # W in Equation (6)
            # nn.init.xavier_uniform_(self.W.weight)
            # nn.init.constant_(self.W.bias, 0)
        elif aggregator_type == 'graphsage':
            self.W = nn.Linear(in_dim * 2, out_dim)   # W in Equation (7)
        elif aggregator_type == 'bi-interaction':
            self.W1 = nn.Linear(in_dim, out_dim)      # W1 in Equation (8)
            self.W2 = nn.Linear(in_dim, out_dim)      # W2 in Equation (8)
        else:
            pass

        self.activation = nn.LeakyReLU()

    def forward(self, g, entity_embed, use_noise=False, show_detail=False):
        # print()
        if 'weight' in g.edata:
            # ma = torch.max(entity_embed).cpu().item()
            # me = torch.mean(entity_embed).cpu().item()
            # v = torch.var(entity_embed).cpu().item()
            # print(f'E_e max: {ma}, mean: {me}, var: {v}')

            g = g.local_var()

            g.ndata['node'] = entity_embed * g.ndata['out_sqrt_degree']
            g.update_all(dgl.function.u_mul_e('node', 'weight', 'side'), dgl.function.sum(msg='side', out='N_h'))
            g.ndata['N_h'] = g.ndata['N_h'] * g.ndata['in_sqrt_degree']

            # g.ndata['node'] = entity_embed
            # g.update_all(dgl.function.u_mul_e('node', 'weight', 'side'), dgl.function.sum(msg='side', out='N_h'))
            # g.ndata['N_h'] = g.ndata['N_h'] * g.ndata['in_sqrt_degree'] * g.ndata['in_sqrt_degree']

            if use_noise:
                # use randn noise with same mean & var
                g.ndata['N_h'] = torch.randn_like(g.ndata['N_h']) * torch.sqrt(g.ndata['N_h'].var()) + g.ndata['N_h'].mean()
            if show_detail:
                print('g.ndata[N_h], mean & abs mean & var:', g.ndata['N_h'].mean().cpu(), g.ndata['N_h'].abs().mean().cpu(), g.ndata['N_h'].var().cpu())
        else:
            g = g.local_var()
            g.ndata['node'] = entity_embed * g.ndata['sqrt_degree']
            g.update_all(dgl.function.copy_src(src='node', out='side'), dgl.function.sum(msg='side', out='N_h'))
            g.ndata['N_h'] = g.ndata['N_h'] * g.ndata['sqrt_degree']

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            out = self.activation(self.W(entity_embed + g.ndata['N_h']))                         # (n_users + n_entities, out_dim)
            # out = self.activation(self.W(g.ndata['N_h']))                                        # have a try

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            out = self.activation(self.W(torch.cat([entity_embed, g.ndata['N_h']], dim=1)))      # (n_users + n_entities, out_dim)

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            out1 = self.activation(self.W1(entity_embed + g.ndata['N_h']))                       # (n_users + n_entities, out_dim)
            out2 = self.activation(self.W2(entity_embed * g.ndata['N_h']))                       # (n_users + n_entities, out_dim)
            out = out1 + out2
        else:
            out = g.ndata['N_h']

        # ma = torch.max(out).cpu().item()
        # me = torch.mean(out).cpu().item()
        # v = torch.var(out).cpu().item()
        # print(f'out max: {ma}, mean: {me}, var: {v}')

        return out


class CFGCN(nn.Module):

    def __init__(self, n_users, n_items, itra_G, struc_Gs=None, embed_dim=64, n_layers=3, lam=0.001, weighted_fuse=False, combine_mode=0, aggregator_type='gcn'):
        super(CFGCN, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.lam = lam
        self.itra_G = itra_G
        self.struc_Gs = struc_Gs
        self.f = nn.Sigmoid()
        self.combine_mode = combine_mode

        self.embedding_user_item_itra = torch.nn.Embedding(num_embeddings=self.n_users + self.n_items, embedding_dim=self.embed_dim)
        nn.init.xavier_uniform_(self.embedding_user_item_itra.weight, gain=1)
        # nn.init.normal_(self.embedding_user_item_itra.weight, std=0.1)
        self.aggregate_layers_itra = []
        for k in range(self.n_layers):
            self.aggregate_layers_itra.append(AggregateUnweighted)
        self.aggregate_layers_itra_p = []
        for k in range(self.n_layers):
            self.aggregate_layers_itra_p.append(AggregateUnweighted_p)
        # self.aggregate_layers_itra_p = self.aggregate_layers_itra

        if self.struc_Gs is not None:
            self.embedding_user_item_struc = self.embedding_user_item_itra
            # self.embedding_user_item_struc = torch.nn.Embedding(num_embeddings=self.n_users + self.n_items, embedding_dim=self.embed_dim)
            # nn.init.xavier_uniform_(self.embedding_user_item_struc.weight, gain=1)
            self.aggregate_layers_struc = []
            for k in range(self.n_layers):
                self.aggregate_layers_struc.append(AggregateWeighted)

            # self.norm_weight = nn.Parameter(torch.ones(1))
            # self.norm_bias = nn.Parameter(torch.zeros(1))
            self.norm_weight_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for i in range(len(struc_Gs))])
            self.norm_bias_list = nn.ParameterList([nn.Parameter(torch.zeros(1)) for i in range(len(struc_Gs))])

        if not weighted_fuse:
            self.layers_weight = None
        else:
            self.layers_weight = nn.ParameterList([nn.Parameter(torch.zeros(1) + 0.3) for i in range(n_layers + 1)])
            nn.init.constant_(self.layers_weight[0], 1)

        # Try to use GCN with parameter
        conv_dim_list = [64, 64, 64, 64]
        aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            aggregator_layers.append(Aggregator(conv_dim_list[k], conv_dim_list[k + 1], aggregator_type))
        self.aggregate_layers_struc = aggregator_layers
        # self.aggregate_layers_itra = aggregator_layers
        # self.aggregate_layers_itra_p = aggregator_layers

    def load_pretrained_embedding(self, pretrained_data):
        assert pretrained_data.shape[0] == self.n_users + self.n_items
        assert pretrained_data.shape[1] == self.embed_dim
        self.embedding_user_item_itra.weight.data = pretrained_data

    def get_pretrained_embedding(self):
        return self.embedding_user_item_itra.weight.data

    def bpr_loss(self, users, pos, neg, use_dummy_gcn=False, use_struc=None):
        if use_dummy_gcn:
            propagate_func = self.dummy_propagate_embedding
        else:
            propagate_func = self.propagate_embedding
        if use_struc is None:
            use_struc = self.struc_Gs is not None

        users_emb_itra_ego = self.embedding_user_item_itra(users.long())
        pos_emb_itra_ego   = self.embedding_user_item_itra(pos.long() + self.n_users)
        neg_emb_itra_ego   = self.embedding_user_item_itra(neg.long() + self.n_users)
        reg_loss = users_emb_itra_ego.norm(2).pow(2) + pos_emb_itra_ego.norm(2).pow(2) + neg_emb_itra_ego.norm(2).pow(2)

        propagated_embed_itra = propagate_func(self.itra_G, self.embedding_user_item_itra, self.aggregate_layers_itra)
        users_emb_itra = propagated_embed_itra[users.long()]
        pos_emb_itra   = propagated_embed_itra[pos.long() + self.n_users]
        neg_emb_itra   = propagated_embed_itra[neg.long() + self.n_users]

        if use_struc:
            assert self.struc_Gs is not None
            users_emb_struc_ego = self.embedding_user_item_struc(users.long())
            pos_emb_struc_ego   = self.embedding_user_item_struc(pos.long() + self.n_users)
            neg_emb_struc_ego   = self.embedding_user_item_struc(neg.long() + self.n_users)
            reg_loss += (users_emb_struc_ego.norm(2).pow(2) + pos_emb_struc_ego.norm(2).pow(2) + neg_emb_struc_ego.norm(2).pow(2))
            # reg_loss = (users_emb_struc_ego.norm(2).pow(2) + pos_emb_struc_ego.norm(2).pow(2) + neg_emb_struc_ego.norm(2).pow(2)) # pure

            users_embs = [users_emb_itra]
            pos_embs = [pos_emb_itra]
            neg_embs = [neg_emb_itra]
            # users_embs = [] # pure
            # pos_embs = [] # pure
            # neg_embs = [] # pure
            for index, g_in in enumerate(self.struc_Gs):
                g = g_in.local_var()
                g.edata['weight'] = g.edata['weight'] * self.norm_weight_list[index] + self.norm_bias_list[index]
                propagated_embed_struc = propagate_func(g, self.embedding_user_item_struc, self.aggregate_layers_struc, use_noise=False)
                users_emb_struc = propagated_embed_struc[users.long()]
                pos_emb_struc   = propagated_embed_struc[pos.long() + self.n_users]
                neg_emb_struc   = propagated_embed_struc[neg.long() + self.n_users]
                users_embs.append(users_emb_struc)
                pos_embs.append(pos_emb_struc)
                neg_embs.append(neg_emb_struc)

            users_emb = combine_multi_graph_embedding(users_embs, mode=self.combine_mode)
            pos_emb = combine_multi_graph_embedding(pos_embs, mode=self.combine_mode)
            neg_emb = combine_multi_graph_embedding(neg_embs, mode=self.combine_mode)
        else:
            users_emb = users_emb_itra
            pos_emb = pos_emb_itra
            neg_emb = neg_emb_itra

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2) * reg_loss / float(len(users))
        return loss + self.lam * reg_loss

    def get_users_ratings(self, users, use_dummy_gcn=False, use_struc=None):
        if use_dummy_gcn:
            propagate_func = self.dummy_propagate_embedding
        else:
            propagate_func = self.propagate_embedding
        if use_struc is None:
            use_struc = self.struc_Gs is not None

        propagated_embed_itra = propagate_func(self.itra_G, self.embedding_user_item_itra, self.aggregate_layers_itra_p)
        users_emb_itra = propagated_embed_itra[users.long()]
        items_emb_itra = propagated_embed_itra[self.n_users:]

        if use_struc:
            assert self.struc_Gs is not None
            users_embs = [users_emb_itra]
            items_embs = [items_emb_itra]
            # users_embs = [] # pure
            # items_embs = [] # pure
            print()
            for index, g_in in enumerate(self.struc_Gs):
                g = g_in.local_var()
                g.edata['weight'] = g.edata['weight'] * self.norm_weight_list[index] + self.norm_bias_list[index]
                print('g.edata[weight] mean & var', g.edata['weight'].mean().cpu(), g.edata['weight'].var().cpu())
                if self.layers_weight is not None:
                    print('self.layers_weight', [w.data.cpu() for w in self.layers_weight])
                # print('embed ego mean & abs mean & var:', self.embedding_user_item_struc.weight.mean().cpu(), self.embedding_user_item_struc.weight.abs().mean().cpu(), self.embedding_user_item_struc.weight.var().cpu())
                propagated_embed_struc = propagate_func(g, self.embedding_user_item_struc, self.aggregate_layers_struc, use_noise=False, show_detail=False)
                users_emb_struc = propagated_embed_struc[users.long()]
                items_emb_struc = propagated_embed_struc[self.n_users:]
                users_embs.append(users_emb_struc)
                items_embs.append(items_emb_struc)

            users_emb = combine_multi_graph_embedding(users_embs, mode=self.combine_mode)
            items_emb = combine_multi_graph_embedding(items_embs, mode=self.combine_mode)
        else:
            users_emb = users_emb_itra
            items_emb = items_emb_itra

        ratings = self.f(torch.matmul(users_emb, items_emb.t()))
        return ratings # shape: (test_batch_size, n_items)


    def dummy_propagate_embedding(self, g_in, ebd_in, agg_layers_in=None, use_noise=False):
        ego_embed = ebd_in(g_in.ndata['id'])
        return ego_embed


    def propagate_embedding(self, g_in, ebd_in, agg_layers_in, use_noise=False, show_detail=False):
        g = g_in.local_var() # try to not use local_var()
        ego_embed = ebd_in(g.ndata['id'])
        all_embed = [ego_embed]

        # print()
        # ma = torch.max(g.edata['weight']).cpu().item()
        # mi = torch.min(g.edata['weight']).cpu().item()
        # me = torch.mean(g.edata['weight']).cpu().item()
        # v = torch.var(g.edata['weight']).cpu().item()
        # print(f'weight max: {ma}, min: {mi}, mean: {me}, var: {v}, edge_nums: {g.number_of_edges()}')

        # ma = torch.max(g.out_degrees().float()).cpu().item()
        # mi = torch.min(g.out_degrees().float()).cpu().item()
        # me = torch.mean(g.out_degrees().float()).cpu().item()
        # v = torch.var(g.out_degrees().float()).cpu().item()
        # print(f'out_degrees max: {ma}, min: {mi}, mean: {me}, var: {v}')
        # print(g.out_degrees().shape, torch.sum(g.out_degrees() == 0))

        # import matplotlib.pyplot as plt
        # # plt.hist(x = g.out_degrees().reshape(-1).cpu().numpy(), range=(0, 99), bins=50, color='steelblue', edgecolor='black')
        # plt.hist(x = g.edata['weight'].reshape(-1).cpu().numpy(), bins=100, color='steelblue', edgecolor='black')
        # plt.show()
        # exit(0)

        for i, layer in enumerate(agg_layers_in):
            ego_embed = layer(g, ego_embed, use_noise, show_detail)
            all_embed.append(ego_embed)

        if self.layers_weight is not None:
            all_embed = [e * self.layers_weight[idx] for idx, e in enumerate(all_embed)]

        # mean version
        all_embed = torch.stack(all_embed, dim=-1)
        propagated_embed = torch.mean(all_embed, dim=-1) # (n_users + n_entities, embed_dim)

        # # only last version
        # propagated_embed = all_embed[-1]

        # # concat version
        # propagated_embed = torch.cat(all_embed, dim=-1)         # (n_users + n_entities, n_layers * embed_dim)

        return propagated_embed


def combine_multi_graph_embedding(embeddings_in, mode=0):
    if mode == 0:
        # mean
        return torch.mean(torch.stack(embeddings_in, dim=-1), dim=-1)
    elif mode == 1:
        # concat
        return torch.cat(embeddings_in, dim=-1)
    else:
        assert False, 'not support this mode in combine_multi_graph_embedding'


# def AggregateUnweighted(g, entity_embed):
#     # try to use a static func instead of a object
#     g = g.local_var()
#     g.ndata['node'] = entity_embed
#     g.update_all(lambda edges: {'side': edges.src['node'] * edges.src['sqrt_degree']},
#                  lambda nodes: {'N_h': nodes.data['sqrt_degree'] * torch.sum(nodes.mailbox['side'], 1)})
#     return g.ndata['N_h']


def AggregateUnweighted_p(g, entity_embed):
    g = g.local_var()
    g.ndata['node'] = entity_embed * g.ndata['sqrt_degree']
    g.update_all(dgl.function.copy_src(src='node', out='side'), lambda nodes: {'N_h': torch.sum(nodes.mailbox['side'], 1)})
    g.ndata['N_h'] = g.ndata['N_h'] * g.ndata['sqrt_degree']
    return g.ndata['N_h']


def AggregateUnweighted(g, entity_embed):
    g = g.local_var()
    g.ndata['node'] = entity_embed * g.ndata['sqrt_degree']
    g.update_all(dgl.function.copy_src(src='node', out='side'), dgl.function.sum(msg='side', out='N_h'))
    g.ndata['N_h'] = g.ndata['N_h'] * g.ndata['sqrt_degree']
    return g.ndata['N_h']


def AggregateWeighted(g, entity_embed):
    g = g.local_var()
    g.ndata['node'] = entity_embed * g.ndata['out_sqrt_degree']
    g.update_all(dgl.function.u_mul_e('node', 'weight', 'side'), dgl.function.sum(msg='side', out='N_h'))
    # g.update_all(lambda edges: {'side' : edges.src['node'] * edges.data['weight']},
    #              lambda nodes: {'N_h': torch.sum(nodes.mailbox['side'], 1)})
    g.ndata['N_h'] = g.ndata['N_h'] * g.ndata['in_sqrt_degree']
    return g.ndata['N_h']


if __name__ == "__main__":
    pass
