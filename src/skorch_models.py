import torch.nn
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW, Adam, SGD
from skorch.callbacks import WandbLogger
from utils.skorch_utils import LearningRateLogger
from tabular.bin.resnet import ResNet, InputShapeSetterResnet
from tabular.bin.mlp import MLP, InputShapeSetterMLP
from tabular.bin.ft_transformer import Transformer


def create_resnet_skorch(id, wandb_run=None, use_checkpoints=True,
                         categorical_indicator=None, **kwargs):
    if "verbose" not in kwargs:
        verbose = 0
    else:
        verbose = kwargs.pop("verbose")
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
    callbacks = [InputShapeSetterResnet(categorical_indicator=categorical_indicator),
                 EarlyStopping(monitor="valid_loss",
                               patience=es_patience)]  # TODO try with train_loss, and in this case use checkpoint
    callbacks.append(EpochScoring(scoring='accuracy', name='train_accuracy', on_train=True))

    if lr_scheduler:
        callbacks.append(LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5,
                                     factor=0.2))  # FIXME make customizable
    if use_checkpoints:
        callbacks.append(Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                                    f_criterion=None))
    if not wandb_run is None:
        callbacks.append(WandbLogger(wandb_run, save_model=False))
        callbacks.append(LearningRateLogger())

    if not categorical_indicator is None:
        categorical_indicator = torch.BoolTensor(categorical_indicator)

    mlp_skorch = NeuralNetClassifier(
        ResNet,
        # Shuffle training data on each epoch
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=optimizer,
        batch_size=max(batch_size, 1),  # if batch size is float, it will be reset during fit
        iterator_train__shuffle=True,
        module__d_numerical=1,  # will be change when fitted
        module__categories=None,  # will be change when fitted
        module__d_out=1,  # idem
        module__regression=False,
        module__categorical_indicator=categorical_indicator,
        verbose=verbose,
        callbacks=callbacks,
        **kwargs
    )

    return mlp_skorch


def create_rtdl_mlp_skorch(id, wandb_run=None, use_checkpoints=True,
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
    callbacks = [InputShapeSetterMLP(),
                 EarlyStopping(monitor="valid_loss",
                               patience=es_patience)]  # TODO try with train_loss, and in this case use checkpoint
    callbacks.append(EpochScoring(scoring='accuracy', name='train_accuracy', on_train=True))
    if lr_scheduler:
        callbacks.append(LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5,
                                     factor=0.2))  # FIXME make customizable
    if use_checkpoints:
        callbacks.append(Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                                    f_criterion=None))
    if not wandb_run is None:
        callbacks.append(WandbLogger(wandb_run, save_model=False))
        callbacks.append(LearningRateLogger())

    if not categorical_indicator is None:
        categorical_indicator = torch.BoolTensor(categorical_indicator)

    mlp_skorch = NeuralNetClassifier(
        MLP,
        # Shuffle training data on each epoch
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=optimizer,
        batch_size=max(batch_size, 1),  # if batch size is float, it will be reset during fit
        iterator_train__shuffle=True,
        module__d_in=1,  # will be change when fitted
        module__categories=None,  # will be change when fitted
        module__d_out=1,  # idem
        module__regression=False,
        module__categorical_indicator=categorical_indicator,
        verbose=0,
        callbacks=callbacks,
        **kwargs
    )

    return mlp_skorch


def create_ft_transformer_skorch(id, wandb_run=None, use_checkpoints=True,
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
    callbacks = [InputShapeSetterResnet(categorical_indicator=categorical_indicator),
                 EarlyStopping(monitor="valid_loss",
                               patience=es_patience)]  # TODO try with train_loss, and in this case use checkpoint
    callbacks.append(EpochScoring(scoring='accuracy', name='train_accuracy', on_train=True))
    if lr_scheduler:
        callbacks.append(LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5,
                                     factor=0.2))  # FIXME make customizable
    if use_checkpoints:
        callbacks.append(Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                                    f_criterion=None))
    if not wandb_run is None:
        callbacks.append(WandbLogger(wandb_run, save_model=False))
        callbacks.append(LearningRateLogger())

    if not categorical_indicator is None:
        categorical_indicator = torch.BoolTensor(categorical_indicator)


    model_skorch = NeuralNetClassifier(
        Transformer,
        # Shuffle training data on each epoch
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=optimizer,
        batch_size=max(batch_size, 1),  # if batch size is float, it will be reset during fit
        iterator_train__shuffle=True,
        module__d_numerical=1,  # will be change when fitted
        module__categories=None,  # will be change when fitted
        module__d_out=1,  # idem
        module__regression=False,
        module__categorical_indicator=categorical_indicator,
        verbose=0,
        callbacks=callbacks,
        **kwargs
    )

    return model_skorch

# from npt.model.npt import NPTModel, InputShapeSetterNPT
# from npt.loss import Loss as NPTLoss

#
# def create_NPT_skorch(id, wandb_run=None, use_checkpoints=True, **kwargs):
#     if "lr_scheduler" not in kwargs:
#         lr_scheduler = False
#     else:
#         lr_scheduler = kwargs.pop("lr_scheduler")
#     if "es_patience" not in kwargs.keys():
#         es_patience = 40
#     else:
#         es_patience = kwargs.pop('es_patience')
#     if "lr_patience" not in kwargs.keys():
#         lr_patience = 30
#     else:
#         lr_patience = kwargs.pop('lr_patience')
#     optimizer = kwargs.pop('optimizer')
#     if optimizer == "adam":
#         optimizer = Adam
#     elif optimizer == "adamw":
#         optimizer = AdamW
#     elif optimizer == "sgd":
#         optimizer = SGD
#     batch_size = kwargs.pop('batch_size')
#     callbacks = [InputShapeSetterNPT(),
#                  EarlyStopping(monitor="valid_loss",
#                                patience=es_patience)]  # TODO try with train_loss, and in this case use checkpoint
#     callbacks.append(EpochScoring(scoring='accuracy', name='train_accuracy', on_train=True))
#
#     if lr_scheduler:
#         callbacks.append(LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5,
#                                      factor=0.2))  # FIXME make customizable
#     if use_checkpoints:
#         callbacks.append(Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
#                                     f_criterion=None))
#     if not wandb_run is None:
#         callbacks.append(WandbLogger(wandb_run, save_model=False))
#         callbacks.append(LearningRateLogger())
#
#     mlp_skorch = NeuralNetClassifier(
#         NPTModel,
#         # Shuffle training data on each epoch
#         criterion=NPTLoss,
#         optimizer=optimizer,
#         batch_size=max(batch_size, 1),  # if batch size is float, it will be reset during fit
#         iterator_train__shuffle=True,
#         module__image_n_patches=False,  # only useful on images
#         criterion__is_minibatch_sgd=True,
#         module__metadata={"input_feature_dims": [],  # will be change when fitted
#                           "cat_features": [],
#                           "num_features": [0]},
#         criterion__metadata={"input_feature_dims": [],  # will be change when fitted
#                              "cat_features": [],
#                              "num_features": [0]},
#         verbose=0,
#         callbacks=callbacks,
#         **kwargs
#     )
#
#     return mlp_skorch
