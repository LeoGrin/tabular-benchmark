from nam.nam.models.nam import NAM
from nam.nam.config.default import defaults

from skorch.callbacks import Checkpoint, LoadInitState, EarlyStopping, LRScheduler
from skorch import NeuralNetClassifier, NeuralNetBinaryClassifier
from torch_models import MLP_npt, MLP_ensemble, InputShapeSetter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import numpy as np
import skorch
import torch.nn as nn


def get_num_units(
        config,
        x):
    num_unique_vals = [len(np.unique(x[:, i])) for i in range(x.shape[1])]
    num_units = [min(config.num_basis_functions, i * config.units_multiplier) for i in num_unique_vals]

    return num_units

class NAMSetter(skorch.callbacks.Callback):
    def __init__(self, config_):
        super(NAMSetter, self).__init__()
        self.config_ = config_
    def on_train_begin(self, net, X, y):
        net.set_params(module__num_inputs=X.shape[1],
                       module__num_units=get_num_units(self.config_, X))



def create_nam_skorch(id, **kwargs):
    config = defaults()
    batch_size = config.batch_size
    mlp_skorch = NeuralNetBinaryClassifier(
        NAM,
        #predict_nonlinearity=None,
        max_epochs=2000,
        lr=0.01,
        # Shuffle training data on each epoch
        optimizer=Adam,
        iterator_train__shuffle=True,
        batch_size=batch_size,
        module__name="nam",
        module__config=config,
        module__num_inputs=1,
        module__num_units=10,
        verbose=0,
        callbacks=[NAMSetter(config),
                   ('lr_scheduler',
                    LRScheduler(policy=ReduceLROnPlateau, patience=10)),
                   EarlyStopping(monitor="train_loss", patience=40),
                   Checkpoint(dirname="skorch_cp", f_params="params_{}.pt".format(id), f_optimizer=None,
                              f_criterion=None)],
        **kwargs
    )
    return mlp_skorch

#>> Config(activation='exu', batch_size=, cross_val=False, data_path='data/GALLUP.csv', decay_rate=0.995, device='cpu', dropout=0.5, early_stopping_patience=50, experiment_name='NAM', feature_dropout=0.5, fold_num=1, hidden_sizes=[64, 32], l2_regularization=0.5, logdir='output', lr=0.0003, num_basis_functions=1000, num_epochs=1, num_folds=5, num_models=1, num_splits=3, num_workers=16, optimizer='adam', output_regularization=0.5, regression=False, save_model_frequency=2, save_top_k=3, seed=2021, shuffle=True, units_multiplier=2, use_dnn=False, wandb=True)
