import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import skorch
import numpy as np


# taken from https://github.com/OATML/non-parametric-transformers/blob/main/baselines/models/dkl_modules.py
# and converted to pytorch lightning
class MLP_npt_lightning(pl.LightningModule):
    def __init__(
            self, input_size, hidden_layer_sizes, output_size,
            dropout_prob=None):
        super().__init__()
        fc_layers = []
        all_layer_sizes = [input_size] + hidden_layer_sizes
        for layer_size_idx in range(len(all_layer_sizes) - 1):
            fc_layers.append(
                nn.Linear(all_layer_sizes[layer_size_idx],
                          all_layer_sizes[layer_size_idx + 1]))

        self.fc_layers = nn.ModuleList(fc_layers)
        self.output_layer = nn.Linear(
            hidden_layer_sizes[-1], output_size)

        if dropout_prob is not None:
            self.dropout = nn.Dropout(p=dropout_prob)
        else:
            self.dropout = None

        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = F.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        output = self.output_layer(x)
        return output

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = self.ce(y_hat, y)
        accuracy = (torch.argmax(y_hat, dim=1) == y) / y.shape[0]
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        # self.log("train_accuracy", accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = self.ce(y_hat, y)
        accuracy = (torch.argmax(y_hat, dim=1) == y) / y.shape[0]
        # Logging to TensorBoard by default
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class InputShapeSetter(skorch.callbacks.Callback):
    def __init__(self, regression=False, batch_size=None):
        self.regression = regression
        self.batch_size = batch_size
    def on_train_begin(self, net, X, y):
        print("here")
        if not net.module_.no_reinitialize:
            print("ha")
            if self.regression:
                net.set_params(module__input_size=X.shape[1], module__output_size=1)
            else:
                net.set_params(module__input_size=X.shape[1], module__output_size=len(np.unique(y)))
            if isinstance(self.batch_size, float):
                train_set_size = min(10000, int(X.shape[0] * 0.75)) #TODO: don't hardcode this
                net.set_params(batch_size=int(self.batch_size * train_set_size))



# class LearningRateLogger(skorch.callbacks.Callback):
#   def on_epoch_end(self, net,
#                       dataset_train=None, dataset_valid=None, **kwargs):
# callbacks = net.callbacks
# for callback in callbacks:
#     if type(callback) == tuple:
#         if callback[0] == 'lr_scheduler':
#             print(callback[1].lr_scheduler_._last_lr)
# print(net.optimizer.param_groups[0]['lr'])
# print(net.optimizer.param_groups[0]['lr'])
class ExU(torch.nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super(ExU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        ## Page(4): initializing the weights using a normal distribution
        ##          N(x; 0:5) with x 2 [3; 4] works well in practice.
        torch.nn.init.trunc_normal_(self.weights, mean=3.0, std=0.5)
        torch.nn.init.trunc_normal_(self.bias, std=0.5)

    def forward(
        self,
        inputs: torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        output = (inputs - self.bias).matmul(torch.exp(self.weights))

        # ReLU activations capped at n (ReLU-n)
        output = F.relu(output)
        output = torch.clamp(output, 0, n)

        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'

# taken from https://github.com/OATML/non-parametric-transformers/blob/main/baselines/models/dkl_modules.py
class MLP_npt(nn.Module):
    def __init__(
            self, input_size, hidden_layer_sizes, output_size,
            dropout_prob=None, softmax=True, resnet=False, activations="relu",
            batchnorm=False, use_exu=False, feature_dropout_prob=None, no_reinitialize=False):
        super().__init__()
        self.no_reinitialize = no_reinitialize
        fc_layers = []
        all_layer_sizes = hidden_layer_sizes
        if resnet:
            for layer_size_idx in range(len(all_layer_sizes) - 1):
                fc_layers.append(nn.Linear(all_layer_sizes[layer_size_idx],
                                           all_layer_sizes[layer_size_idx]))
                fc_layers.append(nn.Linear(all_layer_sizes[layer_size_idx],
                                           all_layer_sizes[layer_size_idx + 1]))
        else:
            for layer_size_idx in range(len(all_layer_sizes) - 1):
                if not use_exu:
                    fc_layers.append(
                        nn.Linear(all_layer_sizes[layer_size_idx],
                                  all_layer_sizes[layer_size_idx + 1]))
                else:
                    fc_layers.append(ExU(all_layer_sizes[layer_size_idx],
                                         all_layer_sizes[layer_size_idx + 1]))

        # self.fc_layers = nn.ModuleList(fc_layers)
        self.fc_layers = nn.ModuleList(fc_layers)
        if not use_exu:
            self.input_layer = nn.Linear(input_size, hidden_layer_sizes[0])
        else:
            self.input_layer = ExU(input_size, hidden_layer_sizes[0])
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], output_size) #last linear layer
        if dropout_prob is not None:
            self.dropout = nn.Dropout(p=dropout_prob)
        else:
            self.dropout = None

        if feature_dropout_prob is not None:
            self.feature_dropout = nn.Dropout(p=feature_dropout_prob)
        else:
            self.feature_dropout = None

        self.batchnorm = batchnorm

        if self.batchnorm:
            self.batchnorm_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in all_layer_sizes])

        self.softmax = softmax
        self.resnet = resnet
        if not use_exu:
            if activations == "relu":
                self.activation = nn.ReLU()
                #self.activation = lambda x:torch.clamp(F.relu(x), 0, 3)
            elif activations == "selu":
                self.activation = nn.SELU()
        else:
            self.activation = nn.Identity()
            print("Using ExU activation, activation parameter ignored")

    def forward(self, x):
        if self.feature_dropout is not None:
            x = self.feature_dropout(x)
        x = self.activation(self.input_layer(x))
        #print('########')
        #print((x == 0).sum() / x.numel())
        #print((x == x.max()).sum() / x.numel())
        if self.dropout is not None:
            x = self.dropout(x)
        if not self.resnet:
            for i, fc_layer in enumerate(self.fc_layers):
                if self.batchnorm:
                    x = self.batchnorm_layers[i](x)
                x = self.activation(fc_layer(x))
                if self.dropout is not None:
                    x = self.dropout(x)
        else:
            for i in range(len(self.fc_layers) // 2):
                old_x = x
                if self.batchnorm:
                    x = self.batchnorm_layers[i](x)
                x = self.activation(self.fc_layers[2 * i](x))
                if self.dropout is not None:
                    x = self.dropout(x)
                x = self.fc_layers[2 * i + 1](x)
                if self.dropout is not None:
                    x = self.dropout(x)
                x = x + old_x
        #print('@@@@@@@@@@@@')
        #print((x == 0).sum() / x.numel())
        #print((x == x.max()).sum() / x.numel())
        output = self.output_layer(x)

        if self.softmax:
            return F.softmax(output, dim=1)
        else:
            return output


class MLP_perso(nn.Module):
    def __init__(self, num_features, num_classes, dropout=0.25, n_hid=128, n_layers=5, batchnorm=False):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.n_hid = n_hid
        self.batchnorm = batchnorm
        self.bias = True
        self.n_layers = n_layers
        self.linear_first = nn.Linear(num_features, n_hid, bias=self.bias)
        self.linears = nn.ModuleList([nn.Linear(n_hid, n_hid, bias=self.bias) for i in range(self.n_layers)])
        self.linear_last = nn.Linear(n_hid, num_classes)
        self.ReLU = nn.ReLU()
        self.Softmax = nn.Softmax(dim=1)
        if self.batchnorm:
            self.batchnorms = nn.ModuleList(
                [nn.BatchNorm1d(n_hid) for i in range(self.n_layers)])

    def forward(self, input_tensor):
        input_tensor = self.linear_first(input_tensor)
        input_tensor = self.ReLU(input_tensor)
        for i in range(self.n_layers):
            input_tensor = self.linears[i](input_tensor)
            if self.batchnorm:
                input_tensor = self.batchnorms[i](input_tensor)
            input_tensor = self.ReLU(input_tensor)
        input_tensor = self.linear_last(input_tensor)
        return self.Softmax(input_tensor)


class MLP_ensemble(nn.Module):
    def __init__(
            self, n_mlps, mlp_size, input_size, hidden_layer_sizes, output_size,
            dropout_prob=None, train_on_different_batch=False):
        super().__init__()
        # assert mlp_size < input_size
        if mlp_size > input_size:
            mlp_size = input_size
        self.n_mlps = n_mlps
        self.train_on_different_batch = train_on_different_batch
        self.output_size = output_size
        mlp_list = []
        self.indice_list = []
        for i in range(n_mlps):
            indices = np.random.choice(range(input_size), mlp_size, replace=False)
            mlp_list.append(MLP_npt(mlp_size, hidden_layer_sizes, output_size,
                                    dropout_prob))  # TODO different output_size for concatenation
            self.indice_list.append(indices)

        self.mlp_list = nn.ModuleList(mlp_list)

    def forward(self, x):
        if self.training and self.train_on_different_batch:
            i = np.random.randint(self.n_mlps)  # TODO
            return self.mlp_list[i](x[:, self.indice_list[i]])
        else:
            votes = torch.zeros((self.n_mlps, x.shape[0], self.output_size))
            for i in range(self.n_mlps):
                votes[i] = self.mlp_list[i](x[:, self.indice_list[i]])
            return torch.mean(votes, axis=0)


class SparseModel(nn.Module):
    def __init__(
            self, input_size, n_layers, n_w, n_concat, output_size, activation_on="sum", hidden_size=None,
            batchnorm=False,
            dropout_prob=None, softmax=True, resnet=False, activations="relu", x_inside=True, temperature=1):
        super().__init__()
        self.activation_on = activation_on
        self.n_concat = n_concat
        self.n_layers = n_layers
        self.n_w = n_w
        if n_concat == 1:
            if not hidden_size is None:
                self.hidden_size = hidden_size
                self.first_layer = nn.Linear(input_size, hidden_size)
            else:
                self.hidden_size = input_size
                self.first_layer = None
        else:
            self.hidden_size = input_size * n_concat
            self.first_layer = None

        if type(temperature) == str:
            if temperature == "identity":
                self.temperature = self.hidden_size
            elif temperature == "sqrt":
                self.temperature = np.sqrt(self.hidden_size)
        else:
            self.temperature = temperature

        self.w_primes = nn.Parameter(torch.empty(n_layers, n_w, self.hidden_size))
        self.ws = nn.Parameter(torch.ones(n_layers, n_w, 1))
        self.output_layer = nn.Linear(
            self.hidden_size, output_size)

        if dropout_prob is not None:
            self.dropout = nn.Dropout(p=dropout_prob)
        else:
            self.dropout = None

        self.softmax = nn.Softmax(dim=1) if softmax else None
        self.softmax_0 = nn.Softmax(dim=0) if softmax else None
        self.resnet = resnet
        if activations == "relu":
            self.activation = nn.ReLU()
        elif activations == "selu":
            self.activation = nn.SELU()
        elif activations == "softmax":
            self.activation = nn.Softmax(dim=1)

        self.x_inside = x_inside
        self.batchnorm = batchnorm
        if batchnorm:
            self.batchnorm_layers = nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for _ in range(self.n_layers)])

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # TODO adapt to real data (correlation?)
        nn.init.normal_(self.w_primes, mean=0, std=1)

    # nn.init.normal(self.ws, mean=1, std=0.1)

    def forward(self, x):
        if self.n_concat > 1:
            x = torch.tile(x, (1, self.n_concat))
        if not (self.first_layer is None):
            x = self.first_layer(x)
        for k in range(self.n_layers):
            if self.batchnorm:
                x = self.batchnorm_layers[k](x)
            x = x / self.temperature
            for i in range(self.n_w):
                if self.x_inside:
                    to_add = self.ws[k][i] * self.softmax(self.w_primes[k][i] * x)
                else:
                    to_add = self.ws[k][i] * x * self.softmax_0(self.w_primes[k][i])
                if self.activation_on == "both" or self.activation_on == "each":
                    to_add = self.activation(to_add)
                x = x + to_add
            if self.activation_on == "both" or self.activation_on == "sum":
                x = self.activation(x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
        return self.softmax(self.output_layer(x))


class SparseModelNew(nn.Module):
    def __init__(
            self, input_size, n_layers, n_hidden, output_size, n_filters_per_hidden_unit=2, linear_output_layer=True,
            batchnorm=False,
            dropout_prob=None, softmax=True, resnet=False, activations="relu", x_inside=False, temperature=1,
    train_selectors = True, bias=False,
    concatenate_input=False,
    train_temperature = False):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_filters_per_hidden_unit = n_filters_per_hidden_unit
        self.input_size = input_size
        self.batchnorm = batchnorm
        self.bias = bias
        self.linear_output_layer = linear_output_layer
        self.concatenate_input = concatenate_input
        self.x_inside  = x_inside
        if concatenate_input:
            self.concatenate_input = concatenate_input
            self.additional_dim = input_size
        else:
            self.additional_dim = 0

        if dropout_prob is not None:
            self.dropout = nn.Dropout(p=dropout_prob)
        else:
            self.dropout = None

        if type(temperature) == str:
            if temperature == "identity":
                temperature = self.hidden_size
            elif temperature == "sqrt":
                temperature = np.sqrt(self.hidden_size)
        if train_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature
        self.w_primes_input =  nn.Parameter(torch.empty(n_filters_per_hidden_unit, n_hidden, input_size))
        self.w_primes = nn.Parameter(
            torch.empty(n_layers, n_filters_per_hidden_unit, n_hidden, n_hidden + self.additional_dim))  # TODO: allow different hiddent sizes
        if not train_selectors:
            self.w_primes_input.requires_grad = False
            self.w_primes.requires_grad = False
        self.ws_input = nn.Parameter(torch.ones(n_filters_per_hidden_unit, n_hidden))
        self.ws = nn.Parameter(torch.ones(n_layers, n_filters_per_hidden_unit, n_hidden))
        if linear_output_layer:
             self.output_layer = nn.Linear(
                 self.n_hidden + self.additional_dim, output_size)
        else:
            self.w_primes_output = nn.Parameter(torch.empty(n_filters_per_hidden_unit, output_size, n_hidden + self.additional_dim))
            self.ws_output = nn.Parameter(torch.ones(n_filters_per_hidden_unit, output_size))

        if bias:
            self.bias_coef_input = nn.Parameter(torch.zeros(n_filters_per_hidden_unit, n_hidden))
            self.bias_coef = nn.Parameter(torch.zeros(n_layers, n_filters_per_hidden_unit, n_hidden))
            if not linear_output_layer:
                self.bias_coef_output = nn.Parameter(torch.zeros(n_filters_per_hidden_unit, output_size))

        self.softmax = nn.Softmax(dim=1) if softmax else None
        self.softmax0 = nn.Softmax(dim=0) if softmax else None
        # self.softmax_0 = nn.Softmax(dim=0) if softmax else None
        # self.resnet = resnet
        # if activations == "relu":
        self.activation = nn.ReLU()
        # elif activations == "selu":
        #     self.activation = nn.SELU()
        # elif activations == "softmax":
        #     self.activation = nn.Softmax(dim=1)
        #
        # self.x_inside = x_inside
        # self.batchnorm = batchnorm
        if batchnorm:
             self.batchnorm_layers = nn.ModuleList([nn.BatchNorm1d(n_hidden + self.additional_dim) for _ in range(self.n_layers)])

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # TODO adapt to real data (correlation?)
        nn.init.normal_(self.w_primes, mean=0, std=1)
        nn.init.normal_(self.w_primes_input, mean=0, std=1)
        nn.init.uniform_(self.ws, -1, 1)
        nn.init.uniform_(self.ws_input, -1, 1)
        if not self.linear_output_layer:
            nn.init.normal_(self.w_primes_output, mean=0, std=1)
            nn.init.uniform_(self.ws_output, -1, 1)
        if self.bias:
            nn.init.normal_(self.bias_coef_input, 0, 1)
            nn.init.normal_(self.bias_coef, 0, 1)
            if not self.linear_output_layer:
                nn.init.normal_(self.bias_coef_output, 0, 1)

    def forward(self, input):
        #for loop version
        # old_x = torch.stack(
        #     [self.activation(
        #         torch.stack(
        #             [self.ws_init[i, j] * (x * self.softmax0(self.w_primes_init[i, j, :])).sum(-1)
        #              for i in range(self.n_filters_per_hidden_unit)], dim=1
        #         ).sum(1)
        #     )
        #         for j in range(self.n_hidden)], dim=1)
        filter_results = []
        for i in range(self.n_filters_per_hidden_unit):
            if self.x_inside:
                filter = self.softmax0(input * self.w_primes_input[i].T / self.temperature)
            else:
                filter = self.softmax0(self.w_primes_input[i].T / self.temperature)
            res = self.ws_input[i] * torch.matmul(input, filter)
            if self.bias:
                res = res + self.bias_coef_input[i]
            filter_results.append(res)
        x = self.activation(torch.stack(filter_results, dim=2).sum(2))
        if self.dropout is not None:
            x = self.dropout(x)


        for k in range(self.n_layers):
            if self.concatenate_input:
                x = torch.cat([x, input], dim=1)
            if self.batchnorm:
                x = self.batchnorm_layers[k](x)
            filter_results = [(self.bias_coef[k, i] if self.bias else 0) + self.ws[k, i] * torch.matmul(x, self.softmax0(self.w_primes[k, i].T / 0.1) / self.temperature) for i in
                              range(self.n_filters_per_hidden_unit)]
            x = self.activation(torch.stack(filter_results, dim=2).sum(2))
            if self.dropout is not None:
                x = self.dropout(x)

        if self.concatenate_input:
            x = torch.cat([x, input], dim=1)

        if self.linear_output_layer:
            self.output_layer(x)
        else:
            filter_results = [(self.bias_coef_output[i] if self.bias else 0) + self.ws_output[i] * torch.matmul(x, self.softmax0(self.w_primes_output[i].T / self.temperature)) for i in range(self.n_filters_per_hidden_unit)]
            x = torch.stack(filter_results, dim=2).sum(2)
        x = self.softmax(x)

        return x


if __name__ == '__main__':
    pass
