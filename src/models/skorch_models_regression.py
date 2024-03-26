import torch.nn
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler
from skorch import NeuralNetRegressor
from skorch.callbacks import EpochScoring
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW, Adam, SGD
from skorch.callbacks import WandbLogger
import sys
sys.path.append("")
from models.tabular.bin.resnet import ResNet, InputShapeSetterResnet
from models.tabular.bin.mlp import MLP, InputShapeSetterMLP
from models.tabular.bin.ft_transformer import Transformer
from models.skorch_models import LearningRateLogger
from models.tabr.bin.tabr import Model, InputShapeSetterTabR
from models.excelFormer.excelFormer import ExcelFormer

class NeuralNetRegressorBis(NeuralNetRegressor):
    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return super().fit(X, y)


def create_excelformer_regressor_skorch(id, wandb_run=None, use_checkpoints=True,
                                   categorical_indicator=None, **kwargs):
    print("excelformer regressor")
    if "lr_scheduler" not in kwargs:
        lr_scheduler = False
    else:
        lr_scheduler = kwargs.pop("lr_scheduler")
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
    batch_size = kwargs.pop('batch_size')
    if "categories" not in kwargs.keys():
        categories = None
    else:
        categories = kwargs.pop('categories')
    callbacks = [InputShapeSetterResnet(regression=True,
                                        categorical_indicator=categorical_indicator,
                                        categories=categories),
                       EarlyStopping(monitor="valid_loss", patience=es_patience)] #TODO try with train_loss, and in this case use checkpoint
    callbacks.append(EpochScoring(scoring='neg_root_mean_squared_error', name='train_accuracy', on_train=True))

    if lr_scheduler:
        callbacks.append(LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)) #FIXME make customizable
    if use_checkpoints:
        callbacks.append(Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                                  f_criterion=None))
    if not wandb_run is None:
        callbacks.append(WandbLogger(wandb_run, save_model=False))
        callbacks.append(LearningRateLogger())
    if not categorical_indicator is None:
        categorical_indicator = torch.BoolTensor(categorical_indicator)

    model = NeuralNetRegressorBis(
        ExcelFormer,
        # Shuffle training data on each epoch
        optimizer=optimizer,
        batch_size=max(batch_size, 1), # if batch size is float, it will be reset during fit
        iterator_train__shuffle=True,
        module__d_numerical=1,  # will be change when fitted
        module__categories=None, # will be change when fitted
        module__d_out=1,  # idem
        #module__categorical_indicator=categorical_indicator,
        callbacks=callbacks,
        #verbose=0,
        **kwargs
    )

    return model

def create_tabr_regressor_skorch(id, wandb_run=None, use_checkpoints=True,
                                   categorical_indicator=None, **kwargs):
    print("resnet regressor")
    if "lr_scheduler" not in kwargs:
        lr_scheduler = False
    else:
        lr_scheduler = kwargs.pop("lr_scheduler")
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
    batch_size = kwargs.pop('batch_size')
    if "categories" not in kwargs.keys():
        categories = None
    else:
        categories = kwargs.pop('categories')
    callbacks = [InputShapeSetterTabR(regression=True,
                                        categorical_indicator=categorical_indicator,
                                        categories=categories,
                                        batch_size=batch_size),
                       EarlyStopping(monitor="valid_loss", patience=es_patience)] #TODO try with train_loss, and in this case use checkpoint
    callbacks.append(EpochScoring(scoring='neg_root_mean_squared_error', name='train_accuracy', on_train=True))

    if lr_scheduler:
        callbacks.append(LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)) #FIXME make customizable
    if use_checkpoints:
        callbacks.append(Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                                  f_criterion=None))
    if not wandb_run is None:
        callbacks.append(WandbLogger(wandb_run, save_model=False))
        callbacks.append(LearningRateLogger())
    if not categorical_indicator is None:
        categorical_indicator = torch.BoolTensor(categorical_indicator)

    model = NeuralNetRegressorBis(
        Model,
        # Shuffle training data on each epoch
        optimizer=optimizer, #TODO weigth decay
        batch_size=max(batch_size, 1) if not type(batch_size) == str else 1, # if batch size is float, it will be reset during fit
        iterator_train__shuffle=True,
        module__n_num_features=1,  # will be change when fitted
        module__n_bin_features=1,  # will be change when fitted
        module__cat_cardinalities=[],  # will be change when fitted
        module__n_classes=1,  # idem
        #module__regression=True,
        module__categorical_indicator=categorical_indicator,
        module__binary_indicator=None, # will be change when fitted
        module__is_train=True,
        callbacks=callbacks,
        **kwargs
    )

    return model


def create_resnet_regressor_skorch(id, wandb_run=None, use_checkpoints=True,
                                   categorical_indicator=None, **kwargs):
    print("resnet regressor")
    if "lr_scheduler" not in kwargs:
        lr_scheduler = False
    else:
        lr_scheduler = kwargs.pop("lr_scheduler")
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
    batch_size = kwargs.pop('batch_size')
    if "categories" not in kwargs.keys():
        categories = None
    else:
        categories = kwargs.pop('categories')
    callbacks = [InputShapeSetterResnet(regression=True,
                                        categorical_indicator=categorical_indicator,
                                        categories=categories),
                       EarlyStopping(monitor="valid_loss", patience=es_patience)] #TODO try with train_loss, and in this case use checkpoint
    callbacks.append(EpochScoring(scoring='neg_root_mean_squared_error', name='train_accuracy', on_train=True))

    if lr_scheduler:
        callbacks.append(LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)) #FIXME make customizable
    if use_checkpoints:
        callbacks.append(Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                                  f_criterion=None))
    if not wandb_run is None:
        callbacks.append(WandbLogger(wandb_run, save_model=False))
        callbacks.append(LearningRateLogger())
    if not categorical_indicator is None:
        categorical_indicator = torch.BoolTensor(categorical_indicator)

    model = NeuralNetRegressorBis(
        ResNet,
        # Shuffle training data on each epoch
        optimizer=optimizer,
        batch_size=max(batch_size, 1), # if batch size is float, it will be reset during fit
        iterator_train__shuffle=True,
        module__d_numerical=1,  # will be change when fitted
        module__categories=None, # will be change when fitted
        module__d_out=1,  # idem
        module__regression=True,
        module__categorical_indicator=categorical_indicator,
        callbacks=callbacks,
        verbose=0,
        **kwargs
    )

    return model

def create_rtdl_mlp_regressor_skorch(id, wandb_run=None, use_checkpoints=True,
                                     categorical_indicator=None, **kwargs):
    if "lr_scheduler" not in kwargs:
        lr_scheduler = False
    else:
        lr_scheduler = kwargs.pop("lr_scheduler")
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
    batch_size = kwargs.pop('batch_size')
    if "categories" not in kwargs.keys():
        categories = None
    else:
        categories = kwargs.pop('categories')
    callbacks = [InputShapeSetterMLP(regression=True,
                                        categorical_indicator=categorical_indicator,
                                        categories=categories),
                       EarlyStopping(monitor="valid_loss", patience=es_patience)] #TODO try with train_loss, and in this case use checkpoint
    callbacks.append(EpochScoring(scoring='neg_root_mean_squared_error', name='train_accuracy', on_train=True))
    if lr_scheduler:
        callbacks.append(LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)) #FIXME make customizable
    if use_checkpoints:
        callbacks.append(Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                                  f_criterion=None))
    if not wandb_run is None:
        callbacks.append(WandbLogger(wandb_run, save_model=False))
        callbacks.append(LearningRateLogger())

    if not categorical_indicator is None:
        categorical_indicator = torch.BoolTensor(categorical_indicator)


    mlp_skorch = NeuralNetRegressorBis(
        MLP,
        # Shuffle training data on each epoch
        optimizer=optimizer,
        batch_size=max(batch_size, 1), # if batch size is float, it will be reset during fit
        iterator_train__shuffle=True,
        module__d_in=1,  # will be change when fitted
        module__categories=None, # will be change when fitted
        module__d_out=1,  # idem
        module__regression=True,
        module__categorical_indicator=categorical_indicator,
        verbose=0,
        callbacks=callbacks,
        **kwargs
    )

    return mlp_skorch

class NeuralNetRegressorCustomOptim(NeuralNetRegressorBis):
    def initialize_optimizer(self, triggered_directly=None):
        """Initialize the model optimizer. If ``self.optimizer__lr``
        is not set, use ``self.lr`` instead.

        Parameters
        ----------
        triggered_directly
          Deprecated, don't use it anymore.

        """
        # handle deprecated paramter
        # if triggered_directly is not None:
        #     warnings.warn(
        #         "The 'triggered_directly' argument to 'initialize_optimizer' is "
        #         "deprecated, please don't use it anymore.", DeprecationWarning)

        named_parameters = list(self.get_all_learnable_params())
        # print
        no_wd_names = ['tokenizer', '.norm', '.bias']
        for x in ['tokenizer', '.norm', '.bias']:
            assert any(x in a for a in (b[0] for b in named_parameters)) #TODO

        def needs_wd(name):
            return all(x not in name for x in no_wd_names)

        named_parameters_grouped = [
            {'params': [v for k, v in named_parameters if needs_wd(k)]},
            {
                'params': [v for k, v in named_parameters if not needs_wd(k)],
                'weight_decay': 0.0,
            }]
        
        args, kwargs = self.get_params_for_optimizer(
            'optimizer', named_parameters)

        # pylint: disable=attribute-defined-outside-init
        self.optimizer_ = self.optimizer(named_parameters_grouped, **kwargs)
        return self

def create_ft_transformer_regressor_skorch(id, wandb_run=None, use_checkpoints=True,
                                           categorical_indicator=None, **kwargs):
    if "lr_scheduler" not in kwargs:
        lr_scheduler = False
    else:
        lr_scheduler = kwargs.pop("lr_scheduler")
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
    batch_size = kwargs.pop('batch_size')
    if "categories" not in kwargs.keys():
        categories = None
    else:
        categories = kwargs.pop('categories')
    callbacks = [InputShapeSetterResnet(regression=True,
                                        categorical_indicator=categorical_indicator,
                                        categories=categories),
                       EarlyStopping(monitor="valid_loss", patience=es_patience)] #TODO try with train_loss, and in this case use checkpoint
    callbacks.append(EpochScoring(scoring='neg_root_mean_squared_error', name='train_accuracy', on_train=True))
    if lr_scheduler:
        callbacks.append(LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)) #FIXME make customizable
    if use_checkpoints:
        callbacks.append(Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                                  f_criterion=None))
    if not wandb_run is None:
        callbacks.append(WandbLogger(wandb_run, save_model=False))
        callbacks.append(LearningRateLogger())

    if not categorical_indicator is None:
        categorical_indicator = torch.BoolTensor(categorical_indicator)


    model_skorch = NeuralNetRegressorCustomOptim(
        Transformer,
        # Shuffle training data on each epoch
        optimizer=optimizer,
        batch_size=max(batch_size, 1), # if batch size is float, it will be reset during fit
        iterator_train__shuffle=True,
        module__d_numerical=1,  # will be change when fitted
        module__categories=None, # will be change when fitted
        module__d_out=1,  # idem
        verbose=0,
        callbacks=callbacks,
        module__regression=True,
        module__categorical_indicator=categorical_indicator,
        **kwargs
    )

    return model_skorch
