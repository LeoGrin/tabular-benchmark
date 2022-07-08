# The SAINT model.
from TabSurvey.models.basemodel_torch import BaseModelTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from torch import einsum
from einops import rearrange

from TabSurvey.models.saint_lib.models.pretrainmodel import SAINT as SAINTModel
from TabSurvey.models.saint_lib.data_openml import DataSetCatCon
from TabSurvey.models.saint_lib.augmentations import embed_data_mask

'''
    SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training
    (https://arxiv.org/abs/2106.01342)
    
    Code adapted from: https://github.com/somepago/saint
'''


class SAINT(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)
        self.model_id = args.model_id
        if len(args.cat_idx) > 0:
            num_idx = list(set(range(args.num_features)) - set(args.cat_idx))
            # Appending 1 for CLS token, this is later used to generate embeddings.
            cat_dims = np.append(np.array([1]), np.array(args.cat_dims)).astype(int)
        else:
            num_idx = list(range(args.num_features))
            cat_dims = np.array([1])

        # Decreasing some hyperparameter to cope with memory issues
        dim = self.params["dim"] if args.num_features < 50 else 8
        self.batch_size = self.args.batch_size if args.num_features < 50 else 64

        print("Using dim %d and batch size %d" % (dim, self.batch_size))

        self.model = SAINTModel(
            categories=tuple(cat_dims),
            num_continuous=len(num_idx),
            dim=dim,
            dim_out=1,
            depth=self.params["depth"],  # 6
            heads=self.params["heads"],  # 8
            attn_dropout=self.params["dropout"],  # 0.1
            ff_dropout=self.params["dropout"],  # 0.1
            mlp_hidden_mults=(4, 2),
            cont_embeddings="MLP",
            attentiontype="colrow",
            final_mlp_style="sep",
            y_dim=args.num_classes
        )

        if self.args.data_parallel:
            self.model.transformer = nn.DataParallel(self.model.transformer, device_ids=self.args.gpu_ids)
            self.model.mlpfory = nn.DataParallel(self.model.mlpfory, device_ids=self.args.gpu_ids)

    def fit(self, X, y, X_val=None, y_val=None):
        # X_val = X[:int(0.2 * len(X))]
        # y_val = y[:int(0.2 * len(y))]
        # X = X[int(0.2 * len(X)):]
        # y = y[int(0.2 * len(y)):]

        if self.args.objective == 'binary':
            criterion = nn.BCEWithLogitsLoss()
        elif self.args.objective == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)#lr=0.00003)

        self.model.to(self.device)

        # SAINT wants it like this...
        X = {'data': X, 'mask': np.ones_like(X)}
        y = {'data': y.reshape(-1, 1)}
        X_val = {'data': X_val, 'mask': np.ones_like(X_val)}
        y_val = {'data': y_val.reshape(-1, 1)}

        train_ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective)
        trainloader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

        val_ds = DataSetCatCon(X_val, y_val, self.args.cat_idx, self.args.objective)
        valloader = DataLoader(val_ds, batch_size=self.args.val_batch_size, shuffle=True, num_workers=4)

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        loss_history = []
        val_loss_history = []

        for epoch in range(self.args.epochs):
            self.model.train()

            for i, data in enumerate(trainloader, 0):
                optimizer.zero_grad()

                # x_categ is the the categorical data,
                # x_cont has continuous data,
                # y_gts has ground truth ys.
                # cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS
                # token) set to 0s.
                # con_mask is an array of ones same shape as x_cont.
                x_categ, x_cont, y_gts, cat_mask, con_mask = data

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

                # We are converting the data to embeddings in the next step
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)

                reps = self.model.transformer(x_categ_enc, x_cont_enc)

                # select only the representations corresponding to CLS token
                # and apply mlp on it in the next step to get the predictions.
                y_reps = reps[:, 0, :]

                y_outs = self.model.mlpfory(y_reps)

                if self.args.objective == "regression":
                    y_gts = y_gts.to(self.device)
                elif self.args.objective == "classification":
                    y_gts = y_gts.to(self.device).squeeze()
                else:
                    y_gts = y_gts.to(self.device).float()

                loss = criterion(y_outs, y_gts)
                loss.backward()
                optimizer.step()

                loss_history.append(loss.item())

                # print("Loss", loss.item())

            # Early Stopping
            val_loss = 0.0
            val_dim = 0
            self.model.eval()
            with torch.no_grad():
                for data in valloader:
                    x_categ, x_cont, y_gts, cat_mask, con_mask = data

                    x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                    cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

                    _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)
                    reps = self.model.transformer(x_categ_enc, x_cont_enc)
                    y_reps = reps[:, 0, :]
                    y_outs = self.model.mlpfory(y_reps)

                    if self.args.objective == "regression":
                        y_gts = y_gts.to(self.device)
                    elif self.args.objective == "classification":
                        y_gts = y_gts.to(self.device).squeeze()
                    else:
                        y_gts = y_gts.to(self.device).float()

                    val_loss += criterion(y_outs, y_gts)
                    val_dim += 1
            val_loss /= val_dim

            val_loss_history.append(val_loss.item())

            print("Epoch", epoch, "loss", val_loss.item())

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_idx = epoch

                # Save the currently best model
                self.save_model(filename_extension="{}_best".format(self.model_id), directory="tmp")

            if min_val_loss_idx + self.args.early_stopping_rounds < epoch:
                print("Validation loss has not improved for %d steps!" % self.args.early_stopping_rounds)
                print("Early stopping applies.")
                break

        self.load_model(filename_extension="{}_best".format(self.model_id), directory="tmp")
        return loss_history, val_loss_history

    def predict_helper(self, X):
        X = {'data': X, 'mask': np.ones_like(X)}
        y = {'data': np.ones((X['data'].shape[0], 1))}

        test_ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective)
        testloader = DataLoader(test_ds, batch_size=self.args.val_batch_size, shuffle=False, num_workers=4)

        self.model.eval()

        predictions = []

        with torch.no_grad():
            for data in testloader:
                x_categ, x_cont, y_gts, cat_mask, con_mask = data

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)
                reps = self.model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:, 0, :]
                y_outs = self.model.mlpfory(y_reps)

                if self.args.objective == "binary":
                    y_outs = torch.sigmoid(y_outs)
                elif self.args.objective == "classification":
                    y_outs = F.softmax(y_outs, dim=1)

                predictions.append(y_outs.detach().cpu().numpy())
        return np.concatenate(predictions)

    def attribute(self, X, y, strategy=""):
        """ Generate feature attributions for the model input.
            Two strategies are supported: default ("") or "diag". The default strategie takes the sum
            over a column of the attention map, while "diag" returns only the diagonal (feature attention to itself)
            of the attention map.
            return array with the same shape as X.
        """
        global my_attention
        # self.load_model(filename_extension="best", directory="tmp")

        X = {'data': X, 'mask': np.ones_like(X)}
        y = {'data': np.ones((X['data'].shape[0], 1))}

        test_ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective)
        testloader = DataLoader(test_ds, batch_size=self.args.val_batch_size, shuffle=False, num_workers=4)

        self.model.eval()
        # print(self.model)
        # Apply hook.
        my_attention = torch.zeros(0)

        def sample_attribution(layer, minput, output):
            global my_attention
            # print(minput)
            """ an hook to extract the attention maps. """
            h = layer.heads
            q, k, v = layer.to_qkv(minput[0]).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
            sim = einsum('b h i d, b h j d -> b h i j', q, k) * layer.scale
            my_attention = sim.softmax(dim=-1)

        # print(type(self.model.transformer.layers[0][0].fn.fn))
        self.model.transformer.layers[0][0].fn.fn.register_forward_hook(sample_attribution)
        attributions = []
        with torch.no_grad():
            for data in testloader:
                x_categ, x_cont, y_gts, cat_mask, con_mask = data

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)
                # print(x_categ.shape, x_cont.shape)
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)
                reps = self.model.transformer(x_categ_enc, x_cont_enc)
                # y_reps = reps[:, 0, :]
                # y_outs = self.model.mlpfory(y_reps)
                if strategy == "diag":
                    attributions.append(my_attention.sum(dim=1)[:, 1:, 1:].diagonal(0, 1, 2))
                else:
                    attributions.append(my_attention.sum(dim=1)[:, 1:, 1:].sum(dim=1))

        attributions = np.concatenate(attributions)
        return attributions

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "dim": trial.suggest_categorical("dim", [32, 64, 128, 256]),
            "depth": trial.suggest_categorical("depth", [1, 2, 3, 6, 12]),
            "heads": trial.suggest_categorical("heads", [2, 4, 8]),
            "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        }
        return params
