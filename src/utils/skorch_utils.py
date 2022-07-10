import skorch
from skorch.callbacks import WandbLogger
import numpy as np


class LearningRateLogger(skorch.callbacks.Callback):
    def on_epoch_begin(self, net,
                       dataset_train=None, dataset_valid=None, **kwargs):
        callbacks = net.callbacks
        for callback in callbacks:
            if isinstance(callback, WandbLogger):
                callback.wandb_run.log({'log_lr': np.log10(net.optimizer_.param_groups[0]['lr'])})

