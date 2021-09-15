"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Felice Antonio Merra'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it, felice.merra@poliba.it'

from abc import ABC

from .FeatureProjection import FeatureProjection
from .DisenGCNLayer import DisenGCNLayer
from collections import OrderedDict

import torch
import torch_geometric
import numpy as np


class DisenGCNModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 weight_size,
                 n_layers,
                 disen_k,
                 temperature,
                 message_dropout,
                 edge_index,
                 random_seed,
                 name="DisenGCN",
                 **kwargs
                 ):
        super().__init__()
        torch.manual_seed(random_seed)

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.weight_size = weight_size
        self.n_layers = n_layers
        self.disen_k = disen_k
        self.temperature = temperature
        self.message_dropout = message_dropout if message_dropout else [0.0] * self.n_layers
        self.weight_size_list = [self.embed_k] + self.weight_size
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64)

        self.Gu = torch.nn.Parameter(
            torch.nn.init.zeros_(torch.empty((self.num_users, self.embed_k))))
        self.Gi = torch.nn.Parameter(
            torch.nn.init.zeros_(torch.empty((self.num_items, self.embed_k))))

        projection_network_list = []

        for layer in range(self.n_layers):
            projection_network_list.append(('feat_proj_' + str(layer), (FeatureProjection(
                self.weight_size_list[layer],
                self.weight_size_list[layer + 1],
                disen_k))))

        self.projection_network = torch.nn.Sequential(OrderedDict(projection_network_list))

        disentangle_network_list = []
        for layer in range(self.n_layers):
            disentangle_network_list.append((DisenGCNLayer(self.message_dropout[layer], self.temperature),
                                             'x, edge_index -> x'))

        self.disentangle_network = torch_geometric.nn.Sequential('x, edge_index', disentangle_network_list)
        self.softplus = torch.nn.Softplus()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, inputs, **kwargs):
        zeta_u = self.projection_network(self.Gu)
        zeta_i = self.projection_network(self.Gi)

        all_zeta = torch.cat((zeta_u, zeta_i), 0)
        all_zeta = self.disentangle_network(all_zeta, self.edge_index)
        zeta_u, zeta_i = torch.split(all_zeta, [self.num_users, self.num_items], 0)

        user, item = inputs

        xui = torch.sum(torch.squeeze(zeta_u[user]) * torch.squeeze(zeta_i[item]), 1)

        return xui, zeta_u, zeta_i

    def predict(self, start, stop, **kwargs):
        return torch.matmul(self.Gu[start:stop], torch.transpose(self.Gi, 0, 1))

    def train_step(self, batch):
        user, pos, neg = batch
        xu_pos, gamma_u, gamma_pos = self.forward(inputs=(user, pos))
        xu_neg, _, gamma_neg = self.forward(inputs=(user, neg))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.sum(self.softplus(-difference))
        reg_loss = self.l_w * (torch.norm(gamma_u, 2) +
                               torch.norm(gamma_pos, 2) +
                               torch.norm(gamma_neg, 2) +
                               torch.stack([torch.norm(value, 2) for value in self.propagation_network.parameters()],
                                           dim=0).sum(dim=0)) * 2
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    @staticmethod
    def get_top_k(preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask), preds, torch.tensor(-np.inf)), k=k, sorted=True)
