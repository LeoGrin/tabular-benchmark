import skorch
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import EpochScoring
from models.torch_models import MLP_npt, MLP_ensemble, InputShapeSetter, SparseModel, SparseModelNew
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.optim import AdamW, Adam, SGD
from skorch.callbacks import WandbLogger
from skorch.scoring import loss_scoring
import numpy as np


class LearningRateLogger(skorch.callbacks.Callback):
    def on_epoch_begin(self, net,
                       dataset_train=None, dataset_valid=None, **kwargs):
        callbacks = net.callbacks
        for callback in callbacks:
            if isinstance(callback, WandbLogger):
                callback.wandb_run.log({'log_lr': np.log10(net.optimizer_.param_groups[0]['lr'])})


class SaveModelHistory(skorch.callbacks.Callback):
    def __init__(self, f=None):
        super(SaveModelHistory, self).__init__()
        self.f = f

    def on_train_end(self, net,
                     dataset_train=None, dataset_valid=None, **kwargs):
        net.save_params(f_params=None,
                        f_optimizer=None,
                        f_history=self.f)


class RevertUpdate(skorch.callbacks.Callback):
    def __init__(self, f_params, f_optimizer):
        super(RevertUpdate, self).__init__()
        self.f_params = f_params
        self.f_optimizer = f_optimizer
        self.n_batches = 5
        self.iter = 0
    def on_train_begin(self, net,
                       X=None, y=None, **kwargs):
        net.save_params(f_params=self.f_params + "_latest",
                        f_optimizer=self.f_optimizer + "_latest")  # to prevent missing file error
        self.iter = 0 #count the number of batch before checking improvement

    def on_batch_begin(self, net, X, y, training, **kwargs):
        # _latest are files corresponding to the latest model
        # We want to compare the model with and without the update at batch n-1 on batch n
        if training:
            self.iter += 1
            if self.iter > 0 and self.iter >= self.n_batches:
                    net.save_params(f_params=self.f_params,
                                    f_optimizer=self.f_optimizer)
                    loss_new = loss_scoring(net, X, y)
                    net.load_params(f_params=self.f_params + "_latest",
                                    f_optimizer=self.f_optimizer + "_latest")
                    loss_old = loss_scoring(net, X, y)
                    if loss_new < loss_old:
                        print("BEST SCORE")
                        # We use the updated model
                        net.load_params(f_params=self.f_params,
                                        f_optimizer=self.f_optimizer)
                        # We replace the checkpoint
                        net.save_params(f_params=self.f_params + "_latest",
                                        f_optimizer=self.f_optimizer + "_latest")
                    self.iter = 0





def create_sparse_model_new_skorch(id, wandb_run, **kwargs):
    if "es_patience" not in kwargs.keys():
        es_patience = 50
    else:
        es_patience = kwargs.pop('es_patience')
    if "lr_patience" not in kwargs.keys():
        lr_patience = 10
    else:
        lr_patience = kwargs.pop('lr_patience')
    if not wandb_run is None:
        mlp_skorch = NeuralNetClassifier(
            SparseModelNew,
            # Shuffle training data on each epoch
            optimizer=Adam,
            iterator_train__shuffle=True,
            module__input_size=1,  # will be change when fitted
            module__output_size=1,  # idem
            verbose=0,
            # LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)
            # LRScheduler(policy=CosineAnnealingWarmRestarts, T_0=50)
            callbacks=[EpochScoring(scoring='accuracy', name='train_acc', on_train=True), LearningRateLogger(),
                       WandbLogger(wandb_run, save_model=False), InputShapeSetter(),
                       ('lr_scheduler',
                        LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)),
                       EarlyStopping(monitor="train_loss", patience=es_patience),
                       Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                                  f_criterion=None),
                       SaveModelHistory(f=r"history/history_{}.json".format(id))],
            **kwargs
        )
    else:
        mlp_skorch = NeuralNetClassifier(
            SparseModelNew,
            # Shuffle training data on each epoch
            optimizer=Adam,
            iterator_train__shuffle=True,
            module__input_size=1,  # will be change when fitted
            module__output_size=1,  # idem
            verbose=100,
            callbacks=[EpochScoring(scoring='accuracy', name='train_acc', on_train=True), InputShapeSetter(),
                       ('lr_scheduler',
                        LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)),
                       EarlyStopping(monitor="train_loss", patience=es_patience),
                       Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                                  f_criterion=None),
                       SaveModelHistory(f=r"history/history_{}.json".format(id))],
            **kwargs
        )
    return mlp_skorch


def create_sparse_model_skorch(id, wandb_run, **kwargs):
    if "es_patience" not in kwargs.keys():
        es_patience = 400
    else:
        es_patience = kwargs.pop('es_patience')
    if "lr_patience" not in kwargs.keys():
        lr_patience = 50
    else:
        lr_patience = kwargs.pop('lr_patience')
    if not wandb_run is None:
        mlp_skorch = NeuralNetClassifier(
            SparseModel,
            # Shuffle training data on each epoch
            optimizer=Adam,
            iterator_train__shuffle=True,
            module__input_size=1,  # will be change when fitted
            module__output_size=1,  # idem
            verbose=0,
            # LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)
            # LRScheduler(policy=CosineAnnealingWarmRestarts, T_0=50)
            callbacks=[EpochScoring(scoring='accuracy', name='train_acc', on_train=True), LearningRateLogger(),
                       WandbLogger(wandb_run, save_model=False), InputShapeSetter(),
                       ('lr_scheduler',
                        LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)),
                       EarlyStopping(monitor="train_loss", patience=es_patience),
                       Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                                  f_criterion=None),
                       SaveModelHistory(f=r"history/history_{}.json".format(id))],
            **kwargs
        )
    else:
        mlp_skorch = NeuralNetClassifier(
            SparseModel,
            # Shuffle training data on each epoch
            optimizer=Adam,
            iterator_train__shuffle=True,
            module__input_size=1,  # will be change when fitted
            module__output_size=1,  # idem
            verbose=0,
            callbacks=[EpochScoring(scoring='accuracy', name='train_acc', on_train=True), InputShapeSetter(),
                       ('lr_scheduler',
                        LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)),
                       EarlyStopping(monitor="train_loss", patience=es_patience),
                       Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                                  f_criterion=None),
                       SaveModelHistory(f=r"history/history_{}.json".format(id))],
            **kwargs
        )
    return mlp_skorch


def create_mlp_skorch(id, wandb_run, **kwargs):
    hidden_size = kwargs.pop('hidden_size')
    n_layers = kwargs.pop('n_layers')
    if "es_patience" not in kwargs.keys():
        es_patience = 40
    else:
        es_patience = kwargs.pop('es_patience')
    if "lr_patience" not in kwargs.keys():
        lr_patience = 30
    else:
        lr_patience = kwargs.pop('lr_patience')
    optimizer = kwargs.pop('optimizer')
    if optimizer == "adam":
        optimizer = Adam
    elif optimizer == "adamw":
        optimizer = AdamW
    if "update_only_if_improve_val" in kwargs.keys():
        update_only_if_improve_val = kwargs.pop('update_only_if_improve_val')
    else:
        update_only_if_improve_val = False
    if "module__input_size" in kwargs.keys():
        module__input_size = kwargs.pop('module__input_size')
    else:
        module__input_size = 1 # will be change when fitted
    if "module__output_size" in kwargs.keys():
        module__output_size = kwargs.pop('module__output_size')
    else:
        module__output_size = 1 # will be change when fitted

    callbacks = [EpochScoring(scoring='accuracy', name='train_acc', on_train=True), InputShapeSetter(),
                 ('lr_scheduler',
                  LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)),
                 EarlyStopping(monitor="train_loss", patience=es_patience),
                 Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                            f_criterion=None),
                 SaveModelHistory(f=r"history/history_{}.json".format(id))]
    if not wandb_run is None:
        callbacks.append(WandbLogger(wandb_run, save_model=False))
        callbacks.append(LearningRateLogger())
    if update_only_if_improve_val:
        print('ACTIVATED')
        f_params = r"params_{}.pt".format(id)
        f_optim = r"optim_{}.pt".format(id)
        dirname = "skorch_cp_updates"
        callbacks.append(RevertUpdate(f_params=dirname + "/" + f_params, f_optimizer=dirname + "/" + f_optim))
    mlp_skorch = NeuralNetClassifier(
        MLP_npt,
        # Shuffle training data on each epoch
        optimizer=optimizer,
        iterator_train__shuffle=True,
        module__hidden_layer_sizes=[hidden_size for i in range(n_layers)],
        module__input_size=module__input_size,  # will be change when fitted
        module__output_size=module__output_size,  # idem
        verbose=0,
        callbacks=callbacks,
        **kwargs
    )
    return mlp_skorch


def create_mlp_skorch_regressor(id, wandb_run, **kwargs):
    hidden_size = kwargs.pop('hidden_size')
    n_layers = kwargs.pop('n_layers')
    if "es_patience" not in kwargs.keys():
        es_patience = 40
    else:
        es_patience = kwargs.pop('es_patience')
    if "lr_patience" not in kwargs.keys():
        lr_patience = 30
    else:
        lr_patience = kwargs.pop('lr_patience')
    optimizer = kwargs.pop('optimizer')
    if optimizer == "adam":
        optimizer = Adam
    elif optimizer == "adamw":
        optimizer = AdamW
    elif optimizer == "sgd":
        optimizer = SGD
    if not wandb_run is None:
        mlp_skorch = NeuralNetRegressor(
            MLP_npt,
            # Shuffle training data on each epoch
            optimizer=optimizer,
            iterator_train__shuffle=True,
            module__hidden_layer_sizes=[hidden_size for i in range(n_layers)],
            module__input_size=1,  # will be change when fitted
            module__output_size=1,  # idem
            module__softmax=False,
            module__no_reinitialize=False,
            verbose=0,
            callbacks=[EpochScoring(scoring='rmse', name='train_acc', on_train=True), LearningRateLogger(),
                       WandbLogger(wandb_run, save_model=False), InputShapeSetter(regression=True),
                      # ('lr_scheduler',
                       # LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)),
                       LRScheduler(policy=CosineAnnealingWarmRestarts, T_0=50),
                       EarlyStopping(monitor="train_loss", patience=es_patience),
                       Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                                  f_criterion=None),
                       SaveModelHistory(f=r"history/history_{}.json".format(id))],
            **kwargs
        )
    else:
        mlp_skorch = NeuralNetRegressor(
            MLP_npt,
            # Shuffle training data on each epoch
            optimizer=optimizer,
            iterator_train__shuffle=True,
            module__hidden_layer_sizes=[hidden_size for i in range(n_layers)],
            module__input_size=1,  # will be change when fitted
            module__output_size=1,  # idem
            module__softmax=False,
            module__no_reinitialize=False,
            verbose=0,
            callbacks=[EpochScoring(scoring='neg_root_mean_squared_error', name='train_rmse', on_train=True),
                       InputShapeSetter(regression=True),
                       ('lr_scheduler',
                        LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)),
                       #LRScheduler(policy=CosineAnnealingWarmRestarts, T_0=50),
                       EarlyStopping(monitor="train_loss", patience=es_patience),
                       Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                                  f_criterion=None),
                       SaveModelHistory(f=r"history/history_{}.json".format(id))],
            **kwargs
        )
    return mlp_skorch


def create_mlp_ensemble_skorch(id, hidden_size, batch_size, n_mlps, mlp_size, train_on_different_batch):
    if train_on_different_batch:
        lr_scheduler_patience = max(10, n_mlps * 2)
        es_patience = max(40, n_mlps * 8)
    else:
        lr_scheduler_patience = 10
        es_patience = 40
    mlp_skorch = NeuralNetClassifier(
        MLP_ensemble,
        max_epochs=2000,
        lr=0.01,
        # Shuffle training data on each epoch
        optimizer=Adam,
        iterator_train__shuffle=True,
        batch_size=batch_size,
        module__hidden_layer_sizes=[hidden_size, hidden_size],
        module__mlp_size=mlp_size,
        module__n_mlps=n_mlps,
        module__train_on_different_batch=train_on_different_batch,
        module__dropout_prob=0.0,
        module__input_size=100,  # will be change when fitted
        module__output_size=1,  # idem
        verbose=0,
        callbacks=[InputShapeSetter(),
                   ('lr_scheduler',
                    LRScheduler(policy=ReduceLROnPlateau, patience=lr_scheduler_patience)),
                   EarlyStopping(monitor="train_loss", patience=es_patience),
                   Checkpoint(dirname="skorch_cp", f_params="params_{}.pt".format(id), f_optimizer=None,
                              f_criterion=None)]
    )
    return mlp_skorch


def mlp_skorch_method(id, hidden_size, batch_size):
    return create_mlp_skorch(id, hidden_size, batch_size)
