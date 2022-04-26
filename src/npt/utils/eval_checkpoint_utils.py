import os
from enum import Enum
from pathlib import Path

import torch
import wandb
from time import sleep
from npt.utils.model_init_utils import (
    init_model_opt_scaler, setup_ddp_model)


class EarlyStopSignal(Enum):
    CONTINUE = 0
    STOP = 1  # Early stopping has triggered
    END = 2  # We have reached the final epoch


class EarlyStopCounter:
    def __init__(self, c, data_cache_prefix, metadata, wandb_run, cv_index,
                 n_splits, device=None):
        """
        :param c: config
        :param data_cache_prefix: str; cache path for the dataset. Used for
            model checkpoints
        :param metadata: Dict, used for model initialization
        :param device: str; set in the distributed setting, otherwise uses
            config option c.exp_device.
        """
        # The number of contiguous epochs for which validation
        # loss has not improved (early stopping)
        self.num_inc_valid_loss_epochs = 0

        # The number of times validation loss has improved since last
        # caching the model -- used for (infrequent) model checkpointing
        self.num_valid_improvements_since_cache = 0

        # The number of times validation loss must improve prior to our
        # caching of the model
        if c.exp_cache_cadence == -1:
            self.cache_cadence = float('inf')  # We will never cache
        else:
            self.cache_cadence = c.exp_cache_cadence

        # Minimum validation loss that the counter has observed
        self.min_val_loss = float('inf')

        self.patience = c.exp_patience
        self.c = c
        self.wandb_run = wandb_run
        self.cv_index = cv_index
        self.n_splits = n_splits

        self.metadata = metadata

        self.stop_signal_message = (
            f'Validation loss has not improved '
            f'for {self.patience} contiguous epochs. '
            f'Stopping evaluation now..')

        # Only needed for distribution
        self.device = device

        # Model caching
        """
        checkpoint_setting: Union[str, None]; have options None,
            best_model, and all_checkpoints.
                None will never checkpoint models.
                best_model will only have in cache at any given time the best
                    performing model yet evaluated.
                all_checkpoints will avoid overwriting, storing each best
                    performing model. Can incur heavy memory load.
        """
        # Cache models to separate directories for each CV split
        #   (for any dataset in which we have multiple splits)
        if self.n_splits > 1:
            data_cache_prefix += f'__cv_{self.cv_index}'

        self.checkpoint_setting = c.exp_checkpoint_setting
        self.model_cache_path = Path(data_cache_prefix) / 'model_checkpoints'
        self.best_model_path = None

        # Only interact with file system in serial mode, or with first GPU
        if self.device is None or self.device == 0:
            # Create cache path, if it doesn't exist
            if not os.path.exists(self.model_cache_path):
                os.makedirs(self.model_cache_path)

            if not self.c.exp_load_from_checkpoint and not self.c.viz_att_maps:
                # Clear cache path, just in case there was a
                # previous run with same config
                self.clear_cache_path()

    def update(
            self, val_loss, model, optimizer, scaler, epoch, end_experiment,
            tradeoff_annealer=None):

        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.num_inc_valid_loss_epochs = 0
            self.num_valid_improvements_since_cache += 1

            # Only cache:
            #   * If not performing a row corr experiment
            #   * If in serial mode, or distributed mode with the GPU0 process
            #   * AND when the validation loss has improved self.cache_cadence
            #       times since the last model caching
            if not self.c.debug_eval_row_interactions:
                if ((self.device is None or self.device == 0) and
                        (self.num_valid_improvements_since_cache >=
                         self.cache_cadence)):
                    print(
                        f'Validation loss has improved '
                        f'{self.num_valid_improvements_since_cache} times since '
                        f'last caching the model. Caching now.')
                    self.cache_model(
                        model=model, optimizer=optimizer, scaler=scaler,
                        val_loss=val_loss, epoch=epoch,
                        tradeoff_annealer=tradeoff_annealer)
                    self.num_valid_improvements_since_cache = 0
        else:
            self.num_inc_valid_loss_epochs += 1

        # Disallow early stopping with patience == -1
        if end_experiment:
            del model
            return EarlyStopSignal.END, self.load_cached_model()
        elif self.patience == -1:
            return EarlyStopSignal.CONTINUE, None
        elif self.num_inc_valid_loss_epochs > self.patience:
            del model
            return EarlyStopSignal.STOP, self.load_cached_model()

        return EarlyStopSignal.CONTINUE, None

    def load_cached_model(self):
        print('\nLoading cached model.')

        # Initialize model and optimizer objects
        model, optimizer, scaler = init_model_opt_scaler(
            self.c, metadata=self.metadata,
            device=self.device)

        # Distribute model, if in distributed setting
        if self.c.mp_distributed:
            model = setup_ddp_model(model=model, c=self.c, device=self.device)

        # Load from checkpoint, populate state dicts
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        # Strict setting -- allows us to load saved attention maps
        # when we wish to visualize them
        model.load_state_dict(checkpoint['model_state_dict'],
                              strict=(not self.c.viz_att_maps))

        if self.c.viz_att_maps:
            optimizer = None
            scaler = None
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(
            f'Successfully loaded cached model from best performing epoch '
            f'{checkpoint["epoch"]}.')

        try:
            num_steps = checkpoint['num_steps']
        except KeyError:
            num_steps = None

        return model, optimizer, scaler, num_steps

    def clear_cache_path(self):
        file_list = [
            f for f in os.listdir(self.model_cache_path)]
        for f in file_list:
            os.remove(self.model_cache_path / f)

    def cache_model(
            self, model, optimizer, scaler,
            val_loss, epoch, tradeoff_annealer=None):
        if self.checkpoint_setting is None:
            return

        if self.checkpoint_setting not in [
                'best_model', 'all_checkpoints']:
            raise NotImplementedError

        # Delete all existing checkpoints
        if self.checkpoint_setting == 'best_model':
            print('Storing new best performing model.')
            self.clear_cache_path()

        val_loss = val_loss.item()
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_loss': val_loss}

        if tradeoff_annealer is not None:
            checkpoint_dict['num_steps'] = tradeoff_annealer.num_steps

        # Store the new model checkpoint
        self.best_model_path = self.model_cache_path / f'model_{epoch}.pt'

        # We encountered issues with the model being reliably checkpointed.
        # This is a clunky way of confirming it is / giving the script
        # "multiple tries", but, if it ain't broke...
        model_is_checkpointed = False
        counter = 0
        while model_is_checkpointed is False and counter < 10000:
            if counter % 10 == 0:
                print(f'Model checkpointing attempts: {counter}.')

            # Attempt to save
            torch.save(checkpoint_dict, self.best_model_path)

            # If we find the file there, continue on
            if os.path.isfile(self.best_model_path):
                model_is_checkpointed = True

            # If the file is not yet found, sleep to avoid bothering the server
            if model_is_checkpointed is False:
                sleep(0.5)

            counter += 1

        # # Save as a wandb artifact
        # artifact = wandb.Artifact(self.c.model_checkpoint_key, type='model')
        # artifact.add_file(str(self.best_model_path))
        # self.wandb_run.log_artifact(artifact)
        # self.wandb_run.join()

        print(
            f'Stored epoch {epoch} model checkpoint to '
            f'{self.best_model_path}.')
        print(f'Val loss: {val_loss}.')

    def is_model_checkpoint(self, file_name):
        return (
            os.path.isfile(self.model_cache_path / file_name) and
            file_name.startswith('model') and file_name.endswith('.pt'))

    @staticmethod
    def get_epoch_from_checkpoint_name(checkpoint_name):
        return int(checkpoint_name.split('.')[0].split('_')[1])

    def get_most_recent_checkpoint(self):
        if not os.path.isdir(self.model_cache_path):
            print(
                f'No cache path yet exists '
                f'{self.model_cache_path}')
            return None

        checkpoint_names = [
            file_or_dir for file_or_dir in os.listdir(self.model_cache_path)
            if self.is_model_checkpoint(file_or_dir)]

        if not checkpoint_names:
            print(
                f'Did not find a checkpoint at cache path '
                f'{self.model_cache_path}')
            return None

        # We assume models stored later are strictly better (i.e. only
        # stored at an improvement in validation loss)
        max_checkpoint_epoch = max(
            [self.get_epoch_from_checkpoint_name(checkpoint_name)
             for checkpoint_name in checkpoint_names])
        self.best_model_path = (
            self.model_cache_path / f'model_{max_checkpoint_epoch}.pt')

        # Return the newest checkpointed model
        return max_checkpoint_epoch, self.load_cached_model()
