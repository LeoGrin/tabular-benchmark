import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import math

import models.deepgbm_lib.config as config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    type_prefix = torch.cuda
else:
    type_prefix = torch


class EmbeddingModel(nn.Module):
    def __init__(self, n_models, max_ntree_per_split, n_output, out_bias=None):
        super(EmbeddingModel, self).__init__()
        self.task = config.config['task']
        self.n_models = n_models
        self.maxleaf = config.config['maxleaf'] + 1
        self.fcs = nn.ModuleList()
        self.max_ntree_per_split = max_ntree_per_split

        embsize = config.config['embsize']

        self.embed_w = Parameter(torch.Tensor(n_models, max_ntree_per_split * self.maxleaf, embsize))
        # torch.nn.init.xavier_normal(self.embed_w)
        stdv = math.sqrt(1.0 / (max_ntree_per_split))
        self.embed_w.data.normal_(0, stdv)  # .uniform_(-stdv, stdv)

        self.bout = BatchDense(n_models, embsize, 1, out_bias)
        self.bn = nn.BatchNorm1d(embsize * n_models)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = torch.nn.Dropout()
        if self.task == 'regression':
            self.criterion = nn.MSELoss()
        elif self.task == 'binary':
            self.criterion = nn.BCELoss()
        else:
            print("Classification task not yet implemented!")
            # TODO: Implement classification

    def batchmul(self, x, models, embed_w, length):
        out = one_hot(x, length)
        out = out.view(x.size(0), models, -1)
        out = out.transpose(0, 1).contiguous()
        out = torch.bmm(out, embed_w)
        out = out.transpose(0, 1).contiguous()
        out = out.view(x.size(0), -1)
        return out

    def lastlayer(self, x):
        out = self.batchmul(x, self.n_models, self.embed_w, self.maxleaf)
        out = self.bn(out)
        # out = self.tanh(out)
        # out = out.view(x.size(0), self.n_models, -1)
        return out

    def forward(self, x):
        out = self.lastlayer(x)
        out = self.dropout(out)
        out = out.view(x.size(0), self.n_models, -1)
        out = self.bout(out)
        # out = self.output_fc(out)
        sum_out = torch.sum(out, -1, True)

        if self.task == 'binary':
            return self.sigmoid(sum_out), out

        # TODO: Implement classification
        return sum_out, out

    def joint_loss(self, out, target, out_inner, target_inner, *args):
        return nn.MSELoss()(out_inner, target_inner)

    def true_loss(self, out, target):
        return self.criterion(out, target)


class BatchDense(nn.Module):
    def __init__(self, batch, in_features, out_features, bias_init=None):
        super(BatchDense, self).__init__()
        self.batch = batch
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(batch, in_features, out_features))
        self.bias = Parameter(torch.Tensor(batch, 1, out_features))
        self.reset_parameters(bias_init)

    def reset_parameters(self, bias_init=None):
        stdv = math.sqrt(6.0 / (self.in_features + self.out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if bias_init is not None:
            self.bias.data = torch.from_numpy(bias_init)
        else:
            self.bias.data.fill_(0)

    def forward(self, x):
        size = x.size()
        # Todo: avoid the swap axis
        x = x.view(x.size(0), self.batch, -1)
        out = x.transpose(0, 1).contiguous()
        out = torch.baddbmm(self.bias, out, self.weight)
        out = out.transpose(0, 1).contiguous()
        out = out.view(x.size(0), -1)
        return out


def one_hot(y, numslot, mask=None):
    y_tensor = y.type(type_prefix.LongTensor).reshape(-1, 1)
    y_one_hot = torch.zeros(y_tensor.size()[0], numslot, device=device, dtype=torch.float32,
                            requires_grad=False).scatter_(1, y_tensor, 1)
    if mask is not None:
        y_one_hot = y_one_hot * mask
    y_one_hot = y_one_hot.view(y.shape[0], -1)
    return y_one_hot
