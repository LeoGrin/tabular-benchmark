import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .models import STGClassificationModel, STGRegressionModel, MLPClassificationModel, MLPRegressionModel, STGCoxModel, MLPCoxModel, L1RegressionModel, SoftThreshRegressionModel, L1GateRegressionModel
from .utils import get_optimizer, as_tensor, as_float, as_numpy, as_cpu, SimpleDataset, FastTensorDataLoader, probe_infnan
from .io import load_state_dict, state_dict
from .meter import GroupMeters
from .losses import calc_concordance_index, PartialLogLikelihood

import logging.config 
import os.path as osp
import time
import numpy as np
import logging
logger = logging.getLogger("my-logger")


__all__ = ['STG']


def _standard_truncnorm_sample(lower_bound, upper_bound, sample_shape=torch.Size()):
    r"""
    Implements accept-reject algorithm for doubly truncated standard normal distribution.
    (Section 2.2. Two-sided truncated normal distribution in [1])
    [1] Robert, Christian P. "Simulation of truncated normal variables." Statistics and computing 5.2 (1995): 121-125.
    Available online: https://arxiv.org/abs/0907.4010
    Args:
        lower_bound (Tensor): lower bound for standard normal distribution. Best to keep it greater than -4.0 for
        stable results
        upper_bound (Tensor): upper bound for standard normal distribution. Best to keep it smaller than 4.0 for
        stable results
    """
    x = torch.randn(sample_shape)
    done = torch.zeros(sample_shape).byte() 
    while not done.all():
        proposed_x = lower_bound + torch.rand(sample_shape) * (upper_bound - lower_bound)
        if (upper_bound * lower_bound).lt(0.0):  # of opposite sign
            log_prob_accept = -0.5 * proposed_x**2
        elif upper_bound < 0.0:  # both negative
            log_prob_accept = 0.5 * (upper_bound**2 - proposed_x**2)
        else:  # both positive
            assert(lower_bound.gt(0.0))
            log_prob_accept = 0.5 * (lower_bound**2 - proposed_x**2)
        prob_accept = torch.exp(log_prob_accept).clamp_(0.0, 1.0)
        accept = torch.bernoulli(prob_accept).byte() & ~done
        if accept.any():
            accept = accept.bool()
            x[accept] = proposed_x[accept]
            accept = accept.byte()
            done |= accept
    return x


class STG(object):
    def __init__(self, device, input_dim=784, output_dim=10, hidden_dims=[400, 200], activation='relu', sigma=0.5, lam=0.1,
                optimizer='Adam', learning_rate=1e-5,  batch_size=100, freeze_onward=None, feature_selection=True, weight_decay=1e-3, 
                task_type='classification', report_maps=False, random_state=1, extra_args=None):
        self.batch_size = batch_size
        self.activation = activation
        self.device = device  # self.get_device(device)
        self.report_maps = report_maps 
        self.task_type = task_type
        self.extra_args = extra_args
        self.freeze_onward = freeze_onward
        self._model = self.build_model(input_dim, output_dim, hidden_dims, activation, sigma, lam, 
                                       task_type, feature_selection)
        self._model.apply(self.init_weights)
        self._model = self._model.to(device)
        self._optimizer = get_optimizer(optimizer, self._model, lr=learning_rate, weight_decay=weight_decay)
    
    def get_device(self, device):
        if device == "cpu":
            device = torch.device("cpu")
        elif device == None:
            args_cuda = torch.cuda.is_available()
            device = device = torch.device("cuda" if args_cuda else "cpu")
        else:
            raise NotImplementedError("Only 'cpu' or 'cuda' is a valid option.")
        return device
        
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            stddev = torch.tensor(0.1)
            shape = m.weight.shape
            m.weight = nn.Parameter(_standard_truncnorm_sample(lower_bound=-2*stddev, upper_bound=2*stddev, 
                                  sample_shape=shape))
            torch.nn.init.zeros_(m.bias)

    def build_model(self, input_dim, output_dim, hidden_dims, activation, sigma, lam, task_type, feature_selection):
        if task_type == 'classification':
            self.metric = nn.CrossEntropyLoss()
            self.tensor_names = ('input','label')
            if feature_selection:
                return STGClassificationModel(input_dim, output_dim, hidden_dims, device=self.device, activation=activation, sigma=sigma, lam=lam)
            else:
                return MLPClassificationModel(input_dim, output_dim, hidden_dims, activation=activation)
        elif task_type == 'regression':
            assert output_dim == 1
            self.metric = nn.MSELoss()
            self.tensor_names = ('input','label')
            if self.extra_args is not None:
                if self.extra_args == 'l1-softthresh':
                    return SoftThreshRegressionModel(input_dim, output_dim, hidden_dims, device=self.device, activation=activation)
                elif self.extra_args == 'l1-norm-reg':
                    return L1RegressionModel(input_dim, output_dim, hidden_dims, device=self.device, activation=activation)
                elif self.extra_args == 'l1-gate':
                    return L1GateRegressionModel(input_dim, output_dim, hidden_dims, device=self.device, activation=activation)
            else:
                if feature_selection:
                    return STGRegressionModel(input_dim, output_dim, hidden_dims, device=self.device, activation=activation, sigma=sigma, lam=lam)
                else:
                    return MLPRegressionModel(input_dim, output_dim, hidden_dims, activation=activation)
        elif task_type == 'cox':
            self.metric = PartialLogLikelihood
            self.tensor_names = ('X', 'E', 'T')
            if feature_selection:
                return STGCoxModel(input_dim, output_dim, hidden_dims, device=self.device, activation=activation, sigma=sigma, lam=lam)
            else:
                return MLPCoxModel(input_dim, output_dim, hidden_dims, activation=activation)
        else:
            raise NotImplementedError()

    def train_step(self, feed_dict, meters=None):
        assert self._model.training

        loss, logits, monitors = self._model(feed_dict)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        #probe_infnan(logits, 'logits')
        if self.task_type=='cox':
            ci = calc_concordance_index(logits.detach().numpy(), 
                    feed_dict['E'].detach().numpy(), feed_dict['T'].detach().numpy())
        if self.extra_args=='l1-softthresh':
            self._model.mlp[0][0].weight.data = self._model.prox_op(self._model.mlp[0][0].weight)

        loss = as_float(loss)
        if meters is not None:
            meters.update(loss=loss)
            if self.task_type =='cox':
                meters.update(CI=ci)
            meters.update(monitors)

    def get_dataloader(self, X, y, shuffle):
        if self.task_type == 'classification':
            data_loader = FastTensorDataLoader(torch.from_numpy(X).float().to(self.device), 
                        torch.from_numpy(y).long().to(self.device), tensor_names=self.tensor_names,
                        batch_size=self.batch_size, shuffle=shuffle)

        elif self.task_type == 'regression':
            data_loader = FastTensorDataLoader(torch.from_numpy(X).float().to(self.device), 
                        torch.from_numpy(y).float().to(self.device), tensor_names=self.tensor_names,
                        batch_size=self.batch_size, shuffle=shuffle)

        elif self.task_type == 'cox':
            assert isinstance(y, dict)
            data_loader = FastTensorDataLoader(torch.from_numpy(X).float().to(self.device), 
                        torch.from_numpy(y['E']).float().to(self.device),
                        torch.from_numpy(y['T']).float().to(self.device),
                        tensor_names=self.tensor_names,
                        batch_size=self.batch_size, shuffle=shuffle)

        else:
            raise NotImplementedError()

        return data_loader 

    def fit(self, X, y, nr_epochs, valid_X=None, valid_y=None, 
        verbose=True, meters=None, early_stop=None, print_interval=1, shuffle=False):
        data_loader = self.get_dataloader(X, y, shuffle)

        if valid_X is not None:
            val_data_loader = self.get_dataloader(valid_X, valid_y, shuffle)
        else:
            val_data_loader = None
        return self.train(data_loader, nr_epochs, val_data_loader, verbose, meters, early_stop, print_interval)

    def evaluate(self, X, y):
        data_loader = self.get_dataloader(X, y, shuffle=None)
        meters = GroupMeters()
        self.validate(data_loader, self.metric, meters, mode='test')
        print(meters.format_simple(''))

    def predict(self, X, verbose=True):
        dataset = SimpleDataset(X)
        data_loader = DataLoader(dataset, batch_size=X.shape[0], shuffle=False) 
        res = []
        self._model.eval()
        for feed_dict in data_loader:
            # feed_dict_np = as_numpy(feed_dict)
            feed_dict = as_tensor(feed_dict)
            feed_dict['input'] = feed_dict['input'].to(self.device)

            with torch.no_grad():
                output_dict = self._model(feed_dict)

            output_dict_np = as_numpy(output_dict)

            # returning probabilities instead of predictions!
            if self.task_type == "classification":
                res.append(output_dict_np['prob'])
            else:
                res.append(output_dict_np['pred'])
        return np.concatenate(res, axis=0)

    def train_epoch(self, data_loader, meters=None):
        if meters is None:
            meters = GroupMeters()

        self._model.train()
        end = time.time()
        for feed_dict in data_loader:
            data_time = time.time() - end; end = time.time()
            self.train_step(feed_dict, meters=meters)
            step_time = time.time() - end; end = time.time()
            #if dev:
            #meters.update({'time/data': data_time, 'time/step': step_time})
        return meters

    def train(self, data_loader, nr_epochs, val_data_loader=None, verbose=True, 
        meters=None, early_stop=None, print_interval=1):

        loss_history = []
        val_loss_history = []

        if meters is None:
            meters = GroupMeters()

        for epoch in range(1, 1 + nr_epochs):
            meters.reset()
            if epoch == self.freeze_onward:
                self._model.freeze_weights()
            self.train_epoch(data_loader, meters=meters)
            loss_history.append(meters.avg['loss'])

            if verbose and epoch % print_interval == 0:
                self.validate(val_data_loader, self.metric, meters)
                caption = 'Epoch: {}:'.format(epoch)
                print(meters.format_simple(caption))

                val_loss_history.append(meters.avg["valid_loss"].item())

            if early_stop is not None:
                flag = early_stop(self._model)
                if flag:
                    break

        return loss_history, val_loss_history

    def validate_step(self, feed_dict, metric, meters=None, mode='valid'):
        with torch.no_grad():
            pred = self._model(feed_dict)
        if self.task_type == 'classification':
            result = metric(pred['logits'], self._model._get_label(feed_dict))
        elif self.task_type == 'regression':
            result = metric(pred['pred'], self._model._get_label(feed_dict))
            
        elif self.task_type == 'cox':
            result = metric(pred['logits'], self._model._get_fail_indicator(feed_dict), 'noties') 
            val_CI = calc_concordance_index(pred['logits'].detach().numpy(), 
                    feed_dict['E'].detach().numpy(), feed_dict['T'].detach().numpy())
            result = as_float(result)
        else:
            raise NotImplementedError()

        if meters is not None:
            meters.update({mode+'_loss':result})
            if self.task_type=='cox':
                meters.update({mode+'_CI':val_CI})

    def validate(self, data_loader, metric, meters=None, mode='valid'):
        if meters is None:
            meters = GroupMeters()

        self._model.eval()
        end = time.time()
        for fd in data_loader:
            data_time = time.time() - end; end = time.time()
            self.validate_step(fd, metric, meters=meters, mode=mode)
            step_time = time.time() - end; end = time.time()

        return meters.avg

    def save_checkpoint(self, filename, extra=None):
        model = self._model

        state = {
            'model': state_dict(model, cpu=True),
            'optimizer': as_cpu(self._optimizer.state_dict()),
            'extra': extra
        }
        try:
            torch.save(state, filename)
            logger.info('Checkpoint saved: "{}".'.format(filename))
        except Exception:
            logger.exception('Error occurred when dump checkpoint "{}".'.format(filename))

    def load_checkpoint(self, filename):
        if osp.isfile(filename):
            model = self._model
            if isinstance(model, nn.DataParallel):
                model = model.module

            try:
                checkpoint = torch.load(filename)
                load_state_dict(model, checkpoint['model'])
                self._optimizer.load_state_dict(checkpoint['optimizer'])
                logger.critical('Checkpoint loaded: {}.'.format(filename))
                return checkpoint['extra']
            except Exception:
                logger.exception('Error occurred when load checkpoint "{}".'.format(filename))
        else:
            logger.warning('No checkpoint found at specified position: "{}".'.format(filename))
        return None

    def get_gates(self, mode):
        return self._model.get_gates(mode)

