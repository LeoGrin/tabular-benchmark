import torch.nn as nn
import torch
import math
import numpy as np
from torch.autograd import Variable


from .layers import MLPLayer, FeatureSelector, GatingLayer
from .losses import PartialLogLikelihood

__all__ = ['MLPModel', 'MLPRegressionModel', 'MLPClassificationModel', 'LinearRegressionModel', 'LinearClassificationModel']


class ModelIOKeysMixin(object):
    def _get_input(self, feed_dict):
        return feed_dict['input']

    def _get_label(self, feed_dict):
        return feed_dict['label']

    def _get_covariate(self, feed_dict):
        '''For cox'''
        return feed_dict['X']

    def _get_fail_indicator(self, feed_dict):
        '''For cox'''
        return feed_dict['E'].reshape(-1, 1)

    def _get_failure_time(self, feed_dict):
        '''For cox'''
        return feed_dict['T']

    def _compose_output(self, value):
        return dict(pred=value)


class MLPModel(MLPLayer):
    def freeze_weights(self):
        for name, p in self.named_parameters():
            if name != 'mu':
                p.requires_grad = False

    def get_gates(self, mode):
        if mode == 'raw':
            return self.mu.detach().cpu().numpy()
        elif mode == 'prob':
            return np.minimum(1.0, np.maximum(0.0, self.mu.detach().cpu().numpy() + 0.5)) 
        else:
            raise NotImplementedError()


class L1RegressionModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, output_dim, hidden_dims, device, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0, lam=0.1):
        super().__init__(input_dim, output_dim, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.loss = nn.MSELoss()
        self.lam = lam

    def forward(self, feed_dict):
        pred = super().forward(self._get_input(feed_dict))
        if self.training:
            loss = self.loss(pred, self._get_label(feed_dict))
            reg = torch.mean(torch.abs(self.mlp[0][0].weight)) 
            total_loss = loss + self.lam * reg
            return total_loss, dict(), dict()
        else:
            return self._compose_output(pred)


class L1GateRegressionModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, output_dim, hidden_dims, device, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0, lam=0.1):
        super().__init__(input_dim, output_dim, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.GateingLayer = GatingLayer(input_dim, device)
        self.reg = self.GateingLayer.regularizer
        self.mu = self.GateingLayer.mu
        self.loss = nn.MSELoss()
        self.lam = lam

    def forward(self, feed_dict):
        x = self.GateingLayer(self._get_input(feed_dict))
        pred = super().forward(x)
        if self.training:
            loss = self.loss(pred, self._get_label(feed_dict))
            reg = torch.mean(self.reg(self.mu))
            total_loss = loss + self.lam * reg
            return total_loss, dict(), dict()
        else:
            return self._compose_output(pred)


class SoftThreshRegressionModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, output_dim, hidden_dims, device, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0, lam=0.1):
        super().__init__(input_dim, output_dim, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.loss = nn.MSELoss()
        self.lam = lam

    def prox_plus(self, w):
        """Projection onto non-negative numbers
        """
        below = w < 0
        w[below] = 0
        return w

    def prox_op(self, w):
        return torch.sign(w) * self.prox_plus(torch.abs(w) - self.lam)

    def forward(self, feed_dict):
        pred = super().forward(self._get_input(feed_dict))
        if self.training:
            loss = self.loss(pred, self._get_label(feed_dict))
            total_loss = loss 
            return total_loss, dict(), dict()
        else:
            return self._compose_output(pred)


class STGRegressionModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, output_dim, hidden_dims, device, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0, lam=0.1):
        super().__init__(input_dim, output_dim, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.FeatureSelector = FeatureSelector(input_dim, sigma, device)
        self.loss = nn.MSELoss()
        self.reg = self.FeatureSelector.regularizer 
        self.lam = lam
        self.mu = self.FeatureSelector.mu
        self.sigma = self.FeatureSelector.sigma

    def forward(self, feed_dict):
        x = self.FeatureSelector(self._get_input(feed_dict))
        pred = super().forward(x)
        if self.training:
            loss = self.loss(pred, self._get_label(feed_dict))
            reg = torch.mean(self.reg((self.mu + 0.5)/self.sigma)) 
            total_loss = loss + self.lam * reg
            return total_loss, dict(), dict()
        else:
            return self._compose_output(pred)
    

class STGClassificationModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, nr_classes, hidden_dims, device, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0, lam=0.1):
        super().__init__(input_dim, nr_classes, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.FeatureSelector = FeatureSelector(input_dim, sigma, device)
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()
        self.reg = self.FeatureSelector.regularizer
        self.lam = lam 
        self.mu = self.FeatureSelector.mu
        self.sigma = self.FeatureSelector.sigma
        
    def forward(self, feed_dict):
        x = self.FeatureSelector(self._get_input(feed_dict))
        logits = super().forward(x)
        if self.training:
            loss = self.loss(logits, self._get_label(feed_dict))
            reg = torch.mean(self.reg((self.mu + 0.5)/self.sigma)) 
            total_loss = loss + self.lam * reg 
            return total_loss, dict(), dict() 
        else:
            return self._compose_output(logits)

    def _compose_output(self, logits):
        value = self.softmax(logits)
        _, pred = value.max(dim=1)
        return dict(prob=value, pred=pred, logits=logits)


class STGCoxModel(MLPModel, ModelIOKeysMixin):
    #TODO: Finish impl cox model.
    def __init__(self, input_dim, nr_classes, hidden_dims, device, lam, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0):
        super().__init__(input_dim, nr_classes, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.FeatureSelector = FeatureSelector(input_dim, sigma, device)
        self.loss = PartialLogLikelihood
        self.noties = 'noties'
        self.lam = lam
        self.reg = self.FeatureSelector.regularizer 
        self.mu = self.FeatureSelector.mu
        self.sigma = self.FeatureSelector.sigma

    def forward(self, feed_dict):
        x = self.FeatureSelector(self._get_covariate(feed_dict))
        logits = super().forward(x)
        if self.training:
            loss = self.loss(logits, self._get_fail_indicator(feed_dict), self.noties)
            reg = torch.sum(self.reg((self.mu + 0.5)/self.sigma)) 
            total_loss = loss + reg 
            return total_loss, logits, dict()
        else:
            return self._compose_output(logits)

    def _compose_output(self, logits):
        return dict(logits=logits)


class MLPCoxModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, nr_classes, hidden_dims, batch_norm=None, dropout=None, activation='relu'):
        super().__init__(input_dim, nr_classes, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.loss = PartialLogLikelihood 
        self.noties = 'noties'

    def forward(self, feed_dict):
        logits = super().forward(self._get_covariate(feed_dict))
        if self.training:
            loss = self.loss(logits, self._get_fail_indicator(feed_dict), self.noties)
            return loss, logits, dict()
        else:
            return self._compose_output(logits)

    def _compose_output(self, logits):
        return dict(logits=logits)


class MLPRegressionModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_norm=None, dropout=None, activation='relu'):
        super().__init__(input_dim, output_dim, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.loss = nn.MSELoss()

    def forward(self, feed_dict):
        pred = super().forward(self._get_input(feed_dict))
        if self.training:
            loss = self.loss(pred, self._get_label(feed_dict))
            return loss, dict(), dict()
        else:
            return self._compose_output(pred)


class MLPClassificationModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, nr_classes, hidden_dims, batch_norm=None, dropout=None, activation='relu'):
        super().__init__(input_dim, nr_classes, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, feed_dict):
        logits = super().forward(self._get_input(feed_dict))
        if self.training:
            loss = self.loss(logits, self._get_label(feed_dict))
            return loss, dict(), dict()
        else:
            return self._compose_output(logits)

    def _compose_output(self, logits):
        value = self.softmax(logits)
        _, pred = value.max(dim=1)
        return dict(prob=value, pred=pred, logits=logits)


class LinearRegressionModel(MLPRegressionModel):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim, [])


class LinearClassificationModel(MLPClassificationModel):
    def __init__(self, input_dim, nr_classes):
        super().__init__(input_dim, nr_classes, [])



    