import time

from models import node_lib
from models.basemodel_torch import BaseModelTorch
from models.node_lib.utils import check_numpy, process_in_chunks

import torch
import torch.nn as nn
import torch.nn.functional as F
from qhoptim.pyt import QHAdam

import numpy as np

from utils.io_utils import get_output_path

'''
    Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data (https://arxiv.org/abs/1909.06312)

    Code adapted from: https://github.com/Qwicen/node
'''


class NODE(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)

        layer_dim = int(self.params["total_tree_count"] / self.params["num_layers"])

        if args.objective == "regression":
            self.model = nn.Sequential(
                node_lib.DenseBlock(args.num_features,
                                    # layer_dim=128, num_layers=8, depth=6, tree_dim=3
                                    layer_dim=layer_dim, num_layers=self.params["num_layers"],
                                    depth=self.params["tree_depth"], tree_dim=self.params["tree_output_dim"],
                                    flatten_output=False,
                                    choice_function=node_lib.entmax15, bin_function=node_lib.entmoid15),
                node_lib.Lambda(lambda x: x[..., 0].mean(dim=-1)),  # average first channels of every tree
            ).to(self.device)

        elif args.objective == "classification" or args.objective == "binary":
            self.model = nn.Sequential(
                node_lib.DenseBlock(args.num_features,
                                    # layer_dim=1024, num_layers=2, depth=6,
                                    layer_dim=layer_dim, num_layers=self.params["num_layers"],
                                    depth=self.params["tree_depth"], tree_dim=args.num_classes + 1,
                                    flatten_output=False,
                                    choice_function=node_lib.entmax15, bin_function=node_lib.entmoid15),
                node_lib.Lambda(lambda x: x[..., :args.num_classes].mean(dim=-2)),
            ).to(self.device)

        print("On:", self.device)

        self.trainer = None
        self.to_device()

    def fit(self, X, y, X_val=None, y_val=None):
        data = node_lib.Dataset(self.args.dataset, random_state=815,
                                X_train=np.array(X, dtype=np.float32), y_train=np.array(y, dtype=np.float32),
                                X_valid=np.array(X_val, dtype=np.float32), y_valid=np.array(y_val, dtype=np.float32))

        with torch.no_grad():
            # trigger data-aware initialisation
            res = self.model(torch.as_tensor(data.X_train[:1000], device=self.device))

        experiment_name = '{}_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}:{:0>2d}'.format(self.args.dataset, *time.gmtime()[:6])

        if self.args.objective == "regression":
            loss_func = F.mse_loss
        elif self.args.objective == "classification":
            loss_func = F.cross_entropy
            data.y_train = data.y_train.astype(int)
        elif self.args.objective == "binary":
            loss_func = F.binary_cross_entropy_with_logits
            data.y_train = data.y_train.reshape(-1, 1)

        self.trainer = node_lib.Trainer(
            model=self.model, loss_function=loss_func,
            experiment_name=experiment_name,
            warm_start=False,
            Optimizer=QHAdam,
            optimizer_params=dict(lr=1e-3, nus=(0.7, 1.0), betas=(0.95, 0.998)),
            verbose=True,
            n_last_checkpoints=5
        )

        best_loss = float('inf')
        best_step_loss = 0

        loss_history = []
        val_loss_history = []

        early_stopping = self.args.early_stopping_rounds + self.args.logging_period

        for batch in node_lib.iterate_minibatches(data.X_train, data.y_train, batch_size=self.args.batch_size, shuffle=True,
                                                  epochs=self.args.epochs):

            metrics = self.trainer.train_on_batch(*batch, device=self.device)
            loss_history.append(metrics['loss'].item())

            if self.trainer.step % self.args.logging_period == 0:
                self.trainer.save_checkpoint()
                self.trainer.average_checkpoints(out_tag='avg')
                self.trainer.load_checkpoint(tag='avg')

                print("Loss %.5f" % (metrics['loss']))

                if self.args.objective == "regression":
                    loss = self.trainer.evaluate_mse(data.X_valid, data.y_valid, device=self.device,
                                                     batch_size=self.args.batch_size)
                    print("Val MSE: %0.5f" % loss)
                elif self.args.objective == "classification":
                    loss = self.trainer.evaluate_logloss(data.X_valid, data.y_valid, device=self.device,
                                                         batch_size=self.args.batch_size)
                    print("Val LogLoss: %0.5f" % loss)
                elif self.args.objective == "binary":
                    loss = self.trainer.evaluate_binarylogloss(data.X_valid, data.y_valid, device=self.device,
                                                               batch_size=self.args.batch_size)

                print("Val Loss: %0.5f" % loss)

                val_loss_history.append(loss)

                if loss < best_loss:
                    best_loss = loss
                    best_step_loss = self.trainer.step
                    self.trainer.save_checkpoint(tag='best')

                self.trainer.load_checkpoint()  # last
                self.trainer.remove_old_temp_checkpoints()

            if self.trainer.step > best_step_loss + early_stopping:
                print('BREAK. There is no improvment for {} steps'.format(early_stopping))
                print("Best step: ", best_step_loss)
                print("Best Val Loss: %0.5f" % best_loss)
                break

        self.trainer.load_checkpoint(tag="best")
        return loss_history, val_loss_history

    def predict_helper(self, X):
        X_test = torch.as_tensor(np.array(X, dtype=np.float), device=self.device, dtype=torch.float32)
        self.model.train(False)
        with torch.no_grad():
            prediction = process_in_chunks(self.model, X_test, batch_size=self.args.val_batch_size)

            if self.args.objective == "classification":
                prediction = F.softmax(prediction, dim=1)
            elif self.args.objective == "binary":
                prediction = torch.sigmoid(prediction)

            prediction = check_numpy(prediction)

        self.predictions = prediction
        return self.predictions

    def save_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="pt")
        print("Saving at", filename)
        self.trainer.save_checkpoint(path=filename)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "num_layers": trial.suggest_categorical("num_layers", [2, 4, 8]),
            "total_tree_count": trial.suggest_categorical("total_tree_count", [1024, 2048]),
            "tree_depth": trial.suggest_categorical("tree_depth", [6, 8]),
            "tree_output_dim": trial.suggest_int("tree_output_dim", 2, 3)
        }
        return params
