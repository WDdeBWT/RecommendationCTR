import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Aggregator(nn.Module):

    def __init__(self):
        super(Aggregator, self).__init__()

    # def forward(self, g, entity_embed):
    #     # try to use a static func instead of a object
    #     g = g.local_var()
    #     g.ndata['node'] = entity_embed

    #     g.update_all(lambda edges: {'side': edges.src['node'] * edges.src['sqrt_degree']},
    #                  lambda nodes: {'N_h': nodes.data['sqrt_degree'] * torch.sum(nodes.mailbox['side'], 1)})
    #     return g.ndata['N_h']

    def forward(self, g, entity_embed):
        # try to use a static func instead of a object
        g = g.local_var()
        g.ndata['node'] = entity_embed * g.ndata['sqrt_degree']
        g.update_all(dgl.function.copy_src(src='node', out='side'), dgl.function.sum(msg='side', out='N_h'))
        g.ndata['N_h'] = g.ndata['N_h'] * g.ndata['sqrt_degree']
        return g.ndata['N_h']


class LightGCN(nn.Module):

    def __init__(self, n_users, n_items, G, embed_dim=64, n_layers=3, lam=0.001):
        super(LightGCN, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.lam = lam
        self.G = G
        self.f = nn.Sigmoid()

        self.embedding_user_item = torch.nn.Embedding(num_embeddings=self.n_users + self.n_items, embedding_dim=self.embed_dim)
        nn.init.xavier_uniform_(self.embedding_user_item.weight, gain=1)

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator())

    def _propagate_embedding(self):
        g = self.G.local_var()
        ego_embed = self.embedding_user_item(g.ndata['id'])
        all_embed = [ego_embed]

        for i, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(g, ego_embed)
            # norm_embed = F.normalize(ego_embed, p=2, dim=1)
            # norm_embed = ego_embed.clone()
            all_embed.append(ego_embed)

        all_embed = torch.stack(all_embed, dim=-1)
        propagated_embed = torch.mean(all_embed, dim=-1) # (n_users + n_entities, embed_dim)
        return propagated_embed

    def bpr_loss(self, users, pos, neg):
        users_emb_ego = self.embedding_user_item(users.long())
        pos_emb_ego   = self.embedding_user_item(pos.long() + self.n_users)
        neg_emb_ego   = self.embedding_user_item(neg.long() + self.n_users)

        propagated_embed = self._propagate_embedding()
        users_emb = propagated_embed[users.long()]
        pos_emb   = propagated_embed[pos.long() + self.n_users]
        neg_emb   = propagated_embed[neg.long() + self.n_users]

        pos_scores = torch.sum(users_emb*pos_emb, dim=1)
        neg_scores = torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2) * (users_emb_ego.norm(2).pow(2) +
                            pos_emb_ego.norm(2).pow(2) +
                            neg_emb_ego.norm(2).pow(2)) / float(len(users))
        return loss + self.lam * reg_loss

    def get_users_ratings(self, users):
        propagated_embed = self._propagate_embedding()
        users_emb = propagated_embed[users.long()]
        items_emb = propagated_embed[self.n_users:]
        ratings = self.f(torch.matmul(users_emb, items_emb.t()))
        return ratings # shape: (test_batch_size, n_items)

if __name__ == "__main__":
    pass
