import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CFGCN(nn.Module):

    def __init__(self, n_users, n_items, itra_G, struc_Gs=None, embed_dim=64, n_layers=3, lam=0.001):
        super(CFGCN, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.lam = lam
        self.itra_G = itra_G
        self.struc_Gs = struc_Gs
        self.f = nn.Sigmoid()

        self.embedding_user_item_itra = torch.nn.Embedding(num_embeddings=self.n_users + self.n_items, embedding_dim=self.embed_dim)
        nn.init.xavier_uniform_(self.embedding_user_item_itra.weight, gain=1)
        self.aggregate_layers_itra = []
        for k in range(self.n_layers):
            self.aggregate_layers_itra.append(AggregateUnweighted)
        self.aggregate_layers_itra_p = []
        for k in range(self.n_layers):
            self.aggregate_layers_itra_p.append(AggregateUnweighted_p)

        if self.struc_Gs is not None:
            self.embedding_user_item_struc = self.embedding_user_item_itra
            # self.embedding_user_item_struc = torch.nn.Embedding(num_embeddings=self.n_users + self.n_items, embedding_dim=self.embed_dim)
            # nn.init.xavier_uniform_(self.embedding_user_item_struc.weight, gain=1)
            self.aggregate_layers_struc = []
            for k in range(self.n_layers):
                self.aggregate_layers_struc.append(AggregateWeighted)

    def bpr_loss(self, users, pos, neg):
        users_emb_itra_ego = self.embedding_user_item_itra(users.long())
        pos_emb_itra_ego   = self.embedding_user_item_itra(pos.long() + self.n_users)
        neg_emb_itra_ego   = self.embedding_user_item_itra(neg.long() + self.n_users)
        reg_loss = users_emb_itra_ego.norm(2).pow(2) + pos_emb_itra_ego.norm(2).pow(2) + neg_emb_itra_ego.norm(2).pow(2)

        propagated_embed_itra = propagate_embedding(self.itra_G, self.embedding_user_item_itra, self.aggregate_layers_itra)
        users_emb_itra = propagated_embed_itra[users.long()]
        pos_emb_itra   = propagated_embed_itra[pos.long() + self.n_users]
        neg_emb_itra   = propagated_embed_itra[neg.long() + self.n_users]

        if self.struc_Gs is not None:
            users_emb_struc_ego = self.embedding_user_item_struc(users.long())
            pos_emb_struc_ego   = self.embedding_user_item_struc(pos.long() + self.n_users)
            neg_emb_struc_ego   = self.embedding_user_item_struc(neg.long() + self.n_users)
            reg_loss += (users_emb_struc_ego.norm(2).pow(2) + pos_emb_struc_ego.norm(2).pow(2) + neg_emb_struc_ego.norm(2).pow(2))

            users_embs = [users_emb_itra]
            pos_embs = [pos_emb_itra]
            neg_embs = [neg_emb_itra]
            for g in self.struc_Gs:
                propagated_embed_struc = propagate_embedding(g, self.embedding_user_item_struc, self.aggregate_layers_struc)
                users_emb_struc = propagated_embed_struc[users.long()]
                pos_emb_struc   = propagated_embed_struc[pos.long() + self.n_users]
                neg_emb_struc   = propagated_embed_struc[neg.long() + self.n_users]
                users_embs.append(users_emb_struc)
                pos_embs.append(pos_emb_struc)
                neg_embs.append(neg_emb_struc)
            users_emb = torch.mean(torch.stack(users_embs, dim=-1), dim=-1)
            pos_emb = torch.mean(torch.stack(pos_embs, dim=-1), dim=-1)
            neg_emb = torch.mean(torch.stack(neg_embs, dim=-1), dim=-1)
        else:
            users_emb = users_emb_itra
            pos_emb = pos_emb_itra
            neg_emb = neg_emb_itra

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2) * reg_loss / float(len(users))
        return loss + self.lam * reg_loss

    def get_users_ratings(self, users):
        propagated_embed_itra = propagate_embedding(self.itra_G, self.embedding_user_item_itra, self.aggregate_layers_itra_p)
        users_emb_itra = propagated_embed_itra[users.long()]
        items_emb_itra = propagated_embed_itra[self.n_users:]

        if self.struc_Gs is not None:
            users_embs = [users_emb_itra]
            items_embs = [items_emb_itra]
            for g in self.struc_Gs:
                propagated_embed_struc = propagate_embedding(g, self.embedding_user_item_struc, self.aggregate_layers_struc)
                users_emb_struc = propagated_embed_struc[users.long()]
                items_emb_struc = propagated_embed_struc[self.n_users:]
                users_embs.append(users_emb_struc)
                items_embs.append(items_emb_struc)
            users_emb = torch.mean(torch.stack(users_embs, dim=-1), dim=-1)
            items_emb = torch.mean(torch.stack(items_embs, dim=-1), dim=-1)
        else:
            users_emb = users_emb_itra
            items_emb = items_emb_itra

        ratings = self.f(torch.matmul(users_emb, items_emb.t()))
        return ratings # shape: (test_batch_size, n_items)


def propagate_embedding(g_in, ebd_in, agg_layers_in):
    g = g_in.local_var()
    ego_embed = ebd_in(g.ndata['id'])
    all_embed = [ego_embed]

    for i, layer in enumerate(agg_layers_in):
        ego_embed = layer(g, ego_embed)
        # norm_embed = F.normalize(ego_embed, p=2, dim=1)
        all_embed.append(ego_embed)

    all_embed = torch.stack(all_embed, dim=-1)
    propagated_embed = torch.mean(all_embed, dim=-1) # (n_users + n_entities, embed_dim)
    return propagated_embed


# def AggregateUnweighted(g, entity_embed):
#     # try to use a static func instead of a object
#     g = g.local_var()
#     g.ndata['node'] = entity_embed
#     g.update_all(lambda edges: {'side': edges.src['node'] * edges.src['sqrt_degree']},
#                  lambda nodes: {'N_h': nodes.data['sqrt_degree'] * torch.sum(nodes.mailbox['side'], 1)})
#     return g.ndata['N_h']


def AggregateUnweighted_p(g, entity_embed):
    # try to use a static func instead of a object
    g = g.local_var()
    g.ndata['node'] = entity_embed * g.ndata['sqrt_degree']
    g.update_all(dgl.function.copy_src(src='node', out='side'), lambda nodes: {'N_h': torch.sum(nodes.mailbox['side'], 1)})
    g.ndata['N_h'] = g.ndata['N_h'] * g.ndata['sqrt_degree']
    return g.ndata['N_h']


def AggregateUnweighted(g, entity_embed):
    # try to use a static func instead of a object
    g = g.local_var()
    g.ndata['node'] = entity_embed * g.ndata['sqrt_degree']
    g.update_all(dgl.function.copy_src(src='node', out='side'), dgl.function.sum(msg='side', out='N_h'))
    g.ndata['N_h'] = g.ndata['N_h'] * g.ndata['sqrt_degree']
    return g.ndata['N_h']


def AggregateWeighted(g, entity_embed):
    # try to use a static func instead of a object
    g = g.local_var()
    g.ndata['node'] = entity_embed * g.ndata['out_sqrt_degree']
    g.update_all(dgl.function.u_mul_e('node', 'weight', 'side'), dgl.function.sum(msg='side', out='N_h'))
    # g.update_all(lambda edges: {'side' : edges.src['node'] * edges.data['weight']},
    #              lambda nodes: {'N_h': torch.sum(nodes.mailbox['side'], 1)})
    g.ndata['N_h'] = g.ndata['N_h'] * g.ndata['in_sqrt_degree']
    return g.ndata['N_h']


if __name__ == "__main__":
    pass
