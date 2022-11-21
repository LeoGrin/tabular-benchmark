from models.basemodel_torch import BaseModelTorch
import os
import numpy as np

from models.danet_lib.DAN_Task import DANetClassifier, DANetRegressor
from models.danet_lib.config.default import cfg
from models.danet_lib.lib.utils import normalize_reg_label

import torch
from qhoptim.pyt import QHAdam

from utils.io_utils import get_output_path

'''
    DANets: Deep Abstract Networks for Tabular Data Classification and Regression (https://arxiv.org/abs/2112.02962)
    
    Code adapted from: https://github.com/WhatAShot/DANet
'''


class DANet(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_ids)
        self.n_gpu = len(args.gpu_ids)

        # No distinction between binary and multiclass classification
        self.task = "regression" if args.objective == "regression" else "classification"
        self.logname = "DANet/" + args.dataset

        # Set model hyperparameters?
        self.model_config = dict(cfg.model)

        self.mu, self.std = None, None

        model_params = {
            "optimizer_fn": QHAdam,
            "optimizer_params": dict(lr=0.008, weight_decay=1e-5, nus=(0.8, 1.0)),
            "scheduler_params": dict(gamma=0.95, step_size=20),
            "scheduler_fn": torch.optim.lr_scheduler.StepLR,
            # "layer": self.model_config['layer'],
            # "base_outdim": self.model_config['base_outdim'],
            # "k": self.model_config['k'],
            # "drop_rate": self.model_config['drop_rate'],
            "seed": cfg.seed,
            **params
        }

        if self.task == "classification":
            self.model = DANetClassifier(**model_params)
            self.eval_metric = ['accuracy']
        else:
            self.model = DANetRegressor(**model_params)
            self.eval_metric = ['mse']

        print(self.model)

    def fit(self, X, y, X_val=None, y_val=None):

        X = np.array(X, dtype=np.float)
        X_val = np.array(X_val, dtype=np.float)

        if self.task == 'regression':
            self.mu, self.std = y.mean(), y.std()
            print("mean = %.5f, std = %.5f" % (self.mu, self.std))
            y = normalize_reg_label(y, self.std, self.mu)
            y_val = normalize_reg_label(y_val, self.std, self.mu)

            # Set Std for Regression Model
            self.model.std = self.std

        self.model.fit(
            X_train=X, y_train=y,
            eval_set=[(X_val, y_val)],
            eval_name=['valid'],
            eval_metric=self.eval_metric,
            max_epochs=self.args.epochs, patience=self.args.early_stopping_rounds,
            batch_size=self.args.batch_size, virtual_batch_size=self.args.batch_size,
            logname=self.logname,
            # resume_dir=self.train_config['resume_dir'],
            n_gpu=self.n_gpu
        )

        return self.model.history["loss"], self.model.history["valid_" + self.eval_metric[0]]

    def predict_helper(self, X):
        assert self.task == "regression"

        X = np.array(X, dtype=np.float)
        preds = self.model.predict(X)

        # How to get the un-normalized prediction? (Regression label is normalized during training)
        preds = preds * self.std + self.mu
        return preds

    def predict_proba(self, X):
        assert self.task == "classification"

        X = np.array(X, dtype=np.float)
        self.prediction_probabilities = self.model.predict_proba(X)
        return self.prediction_probabilities

    def save_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        self.model.save_check(filename)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "layer": trial.suggest_int("layer", 8, 32),
            "base_outdim": trial.suggest_categorical("base_outdim", [64, 96]),
            "k": trial.suggest_int("k", 3, 8),
            "drop_rate": trial.suggest_categorical("drop_rate", [0, 0.1, 0.2, 0.3])
        }
        return params
