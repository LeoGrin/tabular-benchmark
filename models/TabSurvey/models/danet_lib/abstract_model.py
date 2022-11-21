from dataclasses import dataclass, field
from typing import List, Any, Dict
import torch
import torch.cuda
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from qhoptim.pyt import QHAdam
import numpy as np
from abc import abstractmethod
from models.danet_lib.lib.utils import (
    PredictDataset,
    validate_eval_set,
    create_dataloaders,
    define_device,
)
from models.danet_lib.lib.callbacks import (
    CallbackContainer,
    History,
    EarlyStopping,
    LRSchedulerCallback,
)
from models.danet_lib.lib.logger import Train_Log
from models.danet_lib.lib.metrics import MetricContainer, check_metrics
from models.danet_lib.model.DANet import DANet
from models.danet_lib.model.AcceleratedModule import AcceleratedCreator
from sklearn.base import BaseEstimator
from sklearn.utils import check_array

@dataclass
class DANsModel(BaseEstimator):
    """ Class for DANsModel model.
    """
    std: int = None
    drop_rate: float = 0.1
    layer: int = 32
    base_outdim: int = 64
    k: int = 5
    clip_value: int = 2
    seed: int = 1
    verbose: int = 1
    optimizer_fn: Any = QHAdam
    optimizer_params: Dict = field(default_factory=lambda: dict(lr=8e-3, weight_decay=1e-5, nus=(0.8, 1.0)))
    scheduler_fn: Any = torch.optim.lr_scheduler.StepLR
    scheduler_params: Dict = field(default_factory=lambda: dict(gamma=0.95, step_size=20))
    input_dim: int = None
    output_dim: int = None
    device_name: str = "auto"

    def __post_init__(self):
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # Defining device
        self.device = torch.device(define_device(self.device_name))
        if self.verbose != 0:
            print(f"Device used : {self.device}")

    def fit(
        self,
        X_train,
        y_train,
        eval_set=None,
        eval_name=None,
        eval_metric=None,
        loss_fn=None,
        max_epochs=1000,
        patience=500,
        batch_size=8192,
        virtual_batch_size=256,
        callbacks=None,
        logname=None,
        resume_dir=None,
        n_gpu=1
    ):
        """Train a neural network stored in self.network
        Using train_dataloader for training data and
        valid_dataloader for validation.
        Parameters
        ----------
        X_train : np.ndarray
            Train set
        y_train : np.array
            Train targets
        eval_set : list of tuple
            List of eval tuple set (X, y).
            The last one is used for early stopping
        eval_name : list of str
            List of eval set names.
        eval_metric : list of str
            List of evaluation metrics.
            The last metric is used for early stopping.
        loss_fn : callable or None
            a PyTorch loss function
        max_epochs : int
            Maximum number of epochs during training
        patience : int
            Number of consecutive non improving epoch before early stopping
        batch_size : int
            Training batch size
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)
        callbacks : list of callback function
            List of custom callbacks
        logname: str
            Setting log name
        resume_dir: str
            The resume file directory
        gpu_id: str
            Single GPU or Multi GPU ID
        """
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.input_dim = X_train.shape[1]
        self._stop_training = False
        self.log = Train_Log(logname, resume_dir) if (logname or resume_dir) else None
        self.n_gpu = n_gpu
        eval_set = eval_set if eval_set else []

        self.loss_fn = self._default_loss if loss_fn is None else loss_fn
        check_array(X_train)

        self.update_fit_params(X_train, y_train, eval_set)
        # Validate and reformat eval set depending on training data
        eval_names, eval_set = validate_eval_set(eval_set, eval_name, X_train, y_train)
        train_dataloader, valid_dataloaders = self._construct_loaders(X_train, y_train, eval_set)

        self._set_network()
        self._set_metrics(eval_metric, eval_names)
        self._set_optimizer()
        self._set_callbacks(callbacks)

        if resume_dir:
            start_epoch, self.network, self._optimizer, best_value, best_epoch = self.log.load_checkpoint(self._optimizer)


        # Call method on_train_begin for all callbacks
        self._callback_container.on_train_begin()
        best_epoch = 1
        start_epoch = 1
        best_value = -float('inf') if self._task == 'classification' else float('inf')

        print("===> Start training ...")
        for epoch_idx in range(start_epoch, self.max_epochs + 1):
            self.epoch = epoch_idx
            # Call method on_epoch_begin for all callbacks
            self._callback_container.on_epoch_begin(epoch_idx)
            self._train_epoch(train_dataloader)

            # Apply predict epoch to all eval sets
            for eval_name, valid_dataloader in zip(eval_names, valid_dataloaders):
                self._predict_epoch(eval_name, valid_dataloader)

            # Call method on_epoch_end for all callbacks
            self._callback_container.on_epoch_end(epoch_idx, logs=self.history.epoch_metrics)

            #save checkpoint
            self.save_check()
            print('LR: ' + str(self._optimizer.param_groups[0]['lr']))
            if self._stop_training:
                break

        # Call method on_train_end for all callbacks
        self._callback_container.on_train_end()
        self.network.eval()

        return best_value

    def predict(self, X):
        """
        Make predictions on a batch (valid)
        Parameters
        ----------
        X : a :tensor: `torch.Tensor`
            Input data
        Returns
        -------
        predictions : np.array
            Predictions of the regression problem
        """
        self.network.eval()
        dataloader = DataLoader(PredictDataset(X), batch_size=1024, shuffle=False, pin_memory=True)
        results = []
        print('===> Starting test ... ')
        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()
            with torch.no_grad():
                output = self.network(data)
                predictions = output.cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return self.predict_func(res)

    def save_check(self, path=None):
        save_dict = {
            'epoch': self.epoch,
            'model': self.network,
            # 'state_dict': self.network.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'best_value': self._callback_container.callbacks[1].best_loss,
            "best_epoch": self._callback_container.callbacks[1].best_epoch
        }
        if path:
            torch.save(save_dict, path)
        else:
            torch.save(save_dict, self.log.log_dir + '/checkpoint.pth')


    def load_model(self, filepath, input_dim, output_dim, n_gpu=1):
        """Load DANet model.
        Parameters
        ----------
        filepath : str
            Path of the model.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gpu = n_gpu
        load_model = torch.load(filepath, map_location=self.device)
        self.layer, self.virtual_batch_size = load_model['layer_num'], load_model['virtual_batch_size']
        self.k, self.base_outdim = load_model['k'], load_model['base_outdim']
        self._set_network()
        self.network.load_state_dict(load_model['state_dict'])
        self.network.eval()
        accelerated_module = AcceleratedCreator(self.input_dim, base_out_dim=self.base_outdim, k=self.k)
        self.network = accelerated_module(self.network)
        return

    def _train_epoch(self, train_loader):
        """
        Trains one epoch of the network in self.network
        Parameters
        ----------
        train_loader : a :class: `torch.utils.data.Dataloader`
            DataLoader with train set
        """
        self.network.train()
        loss = []
        for batch_idx, (X, y) in enumerate(train_loader):
            self._callback_container.on_batch_begin(batch_idx)
            batch_logs = self._train_batch(X, y)

            self._callback_container.on_batch_end(batch_idx, batch_logs)
            loss.append(batch_logs['loss'])

        epoch_logs = {"lr": self._optimizer.param_groups[-1]["lr"], "loss": np.mean(loss)}

        self.history.epoch_metrics.update(epoch_logs)
        return

    def _train_batch(self, X, y):
        """
        Trains one batch of data
        Parameters
        ----------
        X : torch.Tensor
            Train matrix
        y : torch.Tensor
            Target matrix
        Returns
        -------
        batch_outs : dict
            Dictionnary with "y": target and "score": prediction scores.
        batch_logs : dict
            Dictionnary with "batch_size" and "loss".
        """
        batch_logs = {"batch_size": X.shape[0]}

        X = X.to(self.device).float()
        y = y.to(self.device).float()

        self._optimizer.zero_grad()
        output = self.network(X)
        loss = self.compute_loss(output, y)
        # Perform backward pass and optimization

        loss.backward()
        if self.clip_value:
            clip_grad_norm_(self.network.parameters(), self.clip_value)
        self._optimizer.step()

        batch_logs["loss"] = loss.cpu().detach().numpy().item()

        return batch_logs

    def _predict_epoch(self, name, loader):
        """
        Predict an epoch and update metrics.
        Parameters
        ----------
        name : str
            Name of the validation set
        loader : torch.utils.data.Dataloader
                DataLoader with validation set
        """
        # Setting network on evaluation mode
        self.network.eval()
        list_y_true = []
        list_y_score = []

        # Main loop
        for batch_idx, (X, y) in enumerate(loader):
            scores = self._predict_batch(X)
            list_y_true.append(y)
            list_y_score.append(scores)

        y_true, scores = self.stack_batches(list_y_true, list_y_score)

        metrics_logs = self._metric_container_dict[name](y_true, scores)
        if self._task == 'regression':
            for k, v in metrics_logs.items():
                metrics_logs[k] = v * self.std ** 2
        self.network.train()
        self.history.epoch_metrics.update(metrics_logs)
        return

    def _predict_batch(self, X):
        """
        Predict one batch of data.
        Parameters
        ----------
        X : torch.Tensor
            Owned products
        Returns
        -------
        np.array
            model scores
        """
        X = X.to(self.device).float()

        # compute model output
        with torch.no_grad():
            scores = self.network(X)
            if isinstance(scores, list):
                scores = [x.cpu().detach().numpy() for x in scores]
            else:
                scores = scores.cpu().detach().numpy()

        return scores

    @abstractmethod
    def update_fit_params(self, X_train, y_train, eval_set):
        """
        Set attributes relative to fit function.
        Parameters
        ----------
        X_train : np.ndarray
            Train set
        y_train : np.array
            Train targets
        eval_set : list of tuple
            List of eval tuple set (X, y).
        """
        raise NotImplementedError(
            "users must define update_fit_params to use this base class"
        )

    def _set_network(self):
        """Setup the network and explain matrix."""
        print("===> Building model ...")
        params = {'layer_num': self.layer,
                  'base_outdim': self.base_outdim,
                  'k': self.k,
                  'virtual_batch_size': self.virtual_batch_size,
                  'drop_rate': self.drop_rate,
                  }

        self.network = DANet(self.input_dim, self.output_dim, **params)
        if self.n_gpu > 1 and (self.device == 'cuda' or self.device == torch.device("cuda")):
            self.network = DataParallel(self.network)
        self.network = self.network.to(self.device)

    def _set_metrics(self, metrics, eval_names):
        """Set attributes relative to the metrics.
        Parameters
        ----------
        metrics : list of str
            List of eval metric names.
        eval_names : list of str
            List of eval set names.
        """
        metrics = metrics or [self._default_metric]

        metrics = check_metrics(metrics)
        # Set metric container for each sets
        self._metric_container_dict = {}
        for name in eval_names:
            self._metric_container_dict.update(
                {name: MetricContainer(metrics, prefix=f"{name}_")}
            )

        self._metrics = []
        self._metrics_names = []
        for _, metric_container in self._metric_container_dict.items():
            self._metrics.extend(metric_container.metrics)
            self._metrics_names.extend(metric_container.names)

        # Early stopping metric is the last eval metric

        self.early_stopping_metric = self._metrics_names[-1] if len(self._metrics_names) > 0 else None

    def _set_callbacks(self, custom_callbacks):
        """Setup the callbacks functions.
        Parameters
        ----------
        custom_callbacks : list of func
            List of callback functions.
        """
        # Setup default callbacks history, early stopping and scheduler
        callbacks = []
        self.history = History(self, verbose=self.verbose)
        callbacks.append(self.history)
        if (self.early_stopping_metric is not None) and (self.patience > 0):
            early_stopping = EarlyStopping(
                early_stopping_metric=self.early_stopping_metric,
                is_maximize=self._metrics[-1]._maximize if len(self._metrics) > 0 else None,
                patience=self.patience,
            )
            callbacks.append(early_stopping)
        else:
            print("No early stopping will be performed, last training weights will be used.")

        if self.scheduler_fn is not None:
            # Add LR Scheduler call_back
            is_batch_level = self.scheduler_params.pop("is_batch_level", False)
            scheduler = LRSchedulerCallback(
                scheduler_fn=self.scheduler_fn,
                scheduler_params=self.scheduler_params,
                optimizer=self._optimizer,
                early_stopping_metric=self.early_stopping_metric,
                is_batch_level=is_batch_level,
            )
            callbacks.append(scheduler)

        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        self._callback_container = CallbackContainer(callbacks)
        self._callback_container.set_trainer(self)

    def _set_optimizer(self):
        """Setup optimizer."""
        self._optimizer = self.optimizer_fn(self.network.parameters(), **self.optimizer_params)

    def _construct_loaders(self, X_train, y_train, eval_set):
        """Generate dataloaders for train and eval set.
        Parameters
        ----------
        X_train : np.array
            Train set.
        y_train : np.array
            Train targets.
        eval_set : list of tuple
            List of eval tuple set (X, y).
        Returns
        -------
        train_dataloader : `torch.utils.data.Dataloader`
            Training dataloader.
        valid_dataloaders : list of `torch.utils.data.Dataloader`
            List of validation dataloaders.
        """
        # all weights are not allowed for this type of model
        y_train_mapped = self.prepare_target(y_train)
        for i, (X, y) in enumerate(eval_set):
            y_mapped = self.prepare_target(y)
            eval_set[i] = (X, y_mapped)

        train_dataloader, valid_dataloaders = create_dataloaders(
            X_train,
            y_train_mapped,
            eval_set,
            self.batch_size
        )
        return train_dataloader, valid_dataloaders


    def _update_network_params(self):
        self.network.virtual_batch_size = self.virtual_batch_size

    @abstractmethod
    def compute_loss(self, y_score, y_true):
        """
        Compute the loss.
        Parameters
        ----------
        y_score : a :tensor: `torch.Tensor`
            Score matrix
        y_true : a :tensor: `torch.Tensor`
            Target matrix
        Returns
        -------
        float
            Loss value
        """
        raise NotImplementedError(
            "users must define compute_loss to use this base class"
        )

    @abstractmethod
    def prepare_target(self, y):
        """
        Prepare target before training.
        Parameters
        ----------
        y : a :tensor: `torch.Tensor`
            Target matrix.
        Returns
        -------
        `torch.Tensor`
            Converted target matrix.
        """
        raise NotImplementedError(
            "users must define prepare_target to use this base class"
        )
