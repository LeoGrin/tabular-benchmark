import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models.basemodel_torch import BaseModelTorch

import numpy as np

'''
    TabTransformer: Tabular Data Modeling Using Contextual Embeddings (https://arxiv.org/abs/2012.06678)
    
    Code adapted from: https://github.com/lucidrains/tab-transformer-pytorch
'''


class TabTransformer(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.cat_idx:
            self.num_idx = list(set(range(args.num_features)) - set(args.cat_idx))
            num_continuous = args.num_features - len(args.cat_idx)
            categories_unique = args.cat_dims
        else:
            self.num_idx = list(range(args.num_features))
            num_continuous = args.num_features
            categories_unique = ()
        print(categories_unique)
        print("On Device:", self.device)

        # Decreasing some hyperparameter to cope with memory issues
        dim = self.params["dim"] if args.num_features < 50 else 8
        self.batch_size = self.args.batch_size if args.num_features < 50 else 64

        print("Using dim %d and batch size %d" % (dim, self.batch_size))

        self.model = TabTransformerModel(
            categories=categories_unique,  # tuple (or list?) containing the number of unique values in each category
            num_continuous=num_continuous,  # number of continuous values
            dim_out=args.num_classes,
            mlp_act=nn.ReLU(),  # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            dim=dim,
            depth=self.params["depth"],
            heads=self.params["heads"],
            attn_dropout=self.params["dropout"],
            ff_dropout=self.params["dropout"],
            mlp_hidden_mults=(4, 2)
        )  # .to(self.device)

        self.to_device()

    def fit(self, X, y, X_val=None, y_val=None):
        learning_rate = 10 ** self.params["learning_rate"]
        weight_decay = 10 ** self.params["weight_decay"]
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # For some reason this has to be set explicitly to work with categorical data
        X = np.array(X, dtype=np.float)
        X_val = np.array(X_val, dtype=np.float)

        X = torch.tensor(X).float()
        X_val = torch.tensor(X_val).float()

        y = torch.tensor(y)
        y_val = torch.tensor(y_val)

        if self.args.objective == "regression":
            loss_func = nn.MSELoss()
            y = y.float()
            y_val = y_val.float()
        elif self.args.objective == "classification":
            loss_func = nn.CrossEntropyLoss()
        else:
            loss_func = nn.BCEWithLogitsLoss()
            y = y.float()
            y_val = y_val.float()

        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=2)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.args.val_batch_size, shuffle=True)

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        loss_history = []
        val_loss_history = []

        for epoch in range(self.args.epochs):
            for i, (batch_X, batch_y) in enumerate(train_loader):

                if self.args.cat_idx:
                    x_categ = batch_X[:, self.args.cat_idx].int().to(self.device)
                else:
                    x_categ = None

                x_cont = batch_X[:, self.num_idx].to(self.device)

                out = self.model(x_categ, x_cont)

                if self.args.objective == "regression" or self.args.objective == "binary":
                    out = out.squeeze()

                loss = loss_func(out, batch_y.to(self.device))
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Early Stopping
            val_loss = 0.0
            val_dim = 0
            for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):

                if self.args.cat_idx:
                    x_categ = batch_val_X[:, self.args.cat_idx].int().to(self.device)
                else:
                    x_categ = None

                x_cont = batch_val_X[:, self.num_idx].to(self.device)

                out = self.model(x_categ, x_cont)

                if self.args.objective == "regression" or self.args.objective == "binary":
                    out = out.squeeze()

                val_loss += loss_func(out, batch_val_y.to(self.device))
                val_dim += 1
            val_loss /= val_dim
            val_loss_history.append(val_loss.item())

            print("Epoch %d: Val Loss %.5f" % (epoch, val_loss))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_idx = epoch

                # Save the currently best model
                self.save_model(filename_extension="best", directory="tmp")

            if min_val_loss_idx + self.args.early_stopping_rounds < epoch:
                print("Validation loss has not improved for %d steps!" % self.args.early_stopping_rounds)
                print("Early stopping applies.")
                break

        self.load_model(filename_extension="best", directory="tmp")
        return loss_history, val_loss_history

    def predict_helper(self, X):
        self.model.eval()
        X = np.array(X, dtype=np.float)
        X = torch.tensor(X).float()

        test_dataset = TensorDataset(X)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.val_batch_size, shuffle=False,
                                 num_workers=2)

        predictions = []

        with torch.no_grad():
            for batch_X in test_loader:
                x_categ = batch_X[0][:, self.args.cat_idx].int().to(self.device) if self.args.cat_idx else None
                x_cont = batch_X[0][:, self.num_idx].to(self.device)

                preds = self.model(x_categ, x_cont)

                if self.args.objective == "binary":
                    preds = torch.sigmoid(preds)

                predictions.append(preds.cpu())
        return np.concatenate(predictions)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "dim": trial.suggest_categorical("dim", [32, 64, 128, 256]),  # dimension, paper set at 32
            "depth": trial.suggest_categorical("depth", [1, 2, 3, 6, 12]),  # depth, paper recommended 6
            "heads": trial.suggest_categorical("heads", [2, 4, 8]),  # heads, paper recommends 8
            "weight_decay": trial.suggest_int("weight_decay", -6, -1),  # x = 10 ^ u
            "learning_rate": trial.suggest_int("learning_rate", -6, -3),  # x = 10 ^ u
            "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        }
        return params

    def attribute(self, X: np.ndarray, y: np.ndarray, strategy=""):
        """ Generate feature attributions for the model input.
            Two strategies are supported: default ("") or "diag". The default strategie takes the sum
            over a column of the attention map, while "diag" returns only the diagonal (feature attention to itself)
            of the attention map.
            return array with the same shape as X. The number of columns is equal to the number of categorical values in X.
        """
        X = np.array(X, dtype=np.float)
        # Unroll and Rerun until first attention stage.

        X = torch.tensor(X).float()

        test_dataset = TensorDataset(X)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.args.val_batch_size, shuffle=False,
                                 num_workers=2)

        attentions_list = []
        with torch.no_grad():
            for batch_X in test_loader:
                x_categ = batch_X[0][:, self.args.cat_idx].int().to(self.device) if self.args.cat_idx else None
                x_cont = batch_X[0][:, self.num_idx].to(self.device)
                if x_categ is not None:
                    x_categ += self.model.categories_offset
                    # Tranformer
                    x = self.model.transformer.embeds(x_categ)
                    
                    # Prenorm.
                    x = self.model.transformer.layers[0][0].fn.norm(x)

                    # Attention
                    active_transformer =  self.model.transformer.layers[0][0].fn.fn
                    h = active_transformer.heads
                    q, k, v = active_transformer.to_qkv(x).chunk(3, dim=-1)
                    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
                    sim = einsum('b h i d, b h j d -> b h i j', q, k) * active_transformer.scale
                    attn = sim.softmax(dim=-1) 
                    if strategy == "diag":
                        print(attn.shape)
                        attentions_list.append(attn.diagonal(0,2,3))
                    else:
                        attentions_list.append(attn.sum(dim=1))
                else:
                    raise ValueError("Attention can only be computed for categorical values in TabTransformer.")
            attentions_list = torch.cat(attentions_list).sum(dim=1)
        return attentions_list.numpy()

####################################################################################################################
#
#  TabTransformer code from
#  https://github.com/lucidrains/tab-transformer-pytorch/blob/main/tab_transformer_pytorch/tab_transformer_pytorch.py
#  adapted to work without categorical data
#
#####################################################################################################################
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


# transformer

class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim) # (Embed the categorical features.)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                Residual(PreNorm(dim, FeedForward(dim, dropout=ff_dropout))),
            ]))

    def forward(self, x):
        x = self.embeds(x)

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x


# mlp

class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                self.dim_out = dim_out
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)

        # Added for multiclass output!
        if self.dim_out > 1:
            x = torch.softmax(x, dim=1)
        return x


# main class

class TabTransformerModel(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=2,
            continuous_mean_std=None,
            attn_dropout=0.,
            ff_dropout=0.
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens) # Prepend num_special_tokens.
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        # continuous

        if exists(continuous_mean_std):
            assert continuous_mean_std.shape == (num_continuous,
                                                 2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
        self.register_buffer('continuous_mean_std', continuous_mean_std)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous

        # transformer

        self.transformer = Transformer(
            num_tokens=total_tokens,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )

        # mlp to logits

        input_size = (dim * self.num_categories) + num_continuous
        l = input_size // 8

        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)

    def forward(self, x_categ, x_cont):

        # Adaptation to work without categorical data
        if x_categ is not None:
            assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} ' \
                                                             f'values for your categories input'
            x_categ += self.categories_offset
            x = self.transformer(x_categ)
            flat_categ = x.flatten(1)

        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} ' \
                                                       f'values for your continuous input'

        if exists(self.continuous_mean_std):
            mean, std = self.continuous_mean_std.unbind(dim=-1)
            x_cont = (x_cont - mean) / std

        normed_cont = self.norm(x_cont)

        # Adaptation to work without categorical data 
        if x_categ is not None:
            x = torch.cat((flat_categ, normed_cont), dim=-1)
        else:
            x = normed_cont

        return self.mlp(x)
