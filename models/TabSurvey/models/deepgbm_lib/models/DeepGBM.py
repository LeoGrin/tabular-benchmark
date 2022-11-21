import torch
import torch.nn as nn

from models.deepgbm_lib.models.CatNN import CatNN
from models.deepgbm_lib.models.GBDT2NN import GBDT2NN

import models.deepgbm_lib.config as config

'''

    Define DeepGBM network.
    Combination of CatNN and GBDT2NN.

'''


class DeepGBM(torch.nn.Module):
    def __init__(self, nume_input_size=None, used_features=None,
                 output_w=None, output_b=None,
                 cate_field_size=None, feature_sizes=None,
                 is_shallow_dropout=True, dropout_shallow=[0.5, 0.5],
                 h_depth=2, deep_layers=[32, 32], is_deep_dropout=False,
                 dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation='relu',
                 is_batch_norm=False,
                 func=None):
        super(DeepGBM, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(0.0)) + 1
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.task = config.config['task']

        self.gbdt2nn = GBDT2NN(nume_input_size, used_features, output_w, output_b)

        self.catnn = CatNN(cate_field_size, feature_sizes)

        print('Init DeepGBM succeed!')

        if self.task == 'regression':
            self.criterion = nn.MSELoss()
        elif self.task == 'binary':
            self.criterion = nn.BCELoss()
        elif self.task == 'classification':
            print("Classification not yet implemented")

    def forward(self, Xg, Xd):
        Xd = Xd.long()

        gbdt2nn_out, gbdt2nn_pred = self.gbdt2nn(Xg)
        catnn_out = self.catnn(Xd)

        out = self.alpha * gbdt2nn_out.view(-1) + self.beta * catnn_out.view(-1)

        if self.task == 'binary':
            return nn.Sigmoid()(out), gbdt2nn_pred

        return out, gbdt2nn_pred

    def joint_loss(self, out, target, gbdt2nn_emb_pred, gbdt2nn_emb_target, ratio):
        # true_loss = (1-ratio) * self.criterion(out.view(-1), target.view(-1))
        return (1 - ratio) * self.true_loss(out, target) + ratio * self.gbdt2nn.emb_loss(gbdt2nn_emb_pred,
                                                                                         gbdt2nn_emb_target)

    def true_loss(self, out, target):
        return self.criterion(out.view(-1), target.view(-1))
