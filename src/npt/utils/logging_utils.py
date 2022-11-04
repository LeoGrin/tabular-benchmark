"""Some logging utils."""

import time
import torch
import wandb


class Logger:
    def __init__(self, c, optimizer, gpu, tradeoff_annealer):
        self.c = c
        self.optimizer = optimizer
        self.gpu = gpu  # If not None, only log for GPU 0
        self.tradeoff_annealer = tradeoff_annealer

    def start_counting(self):
        self.train_start = time.time()
        self.checkpoint_start = self.train_start

    def log(self, train_loss, val_loss, test_loss, steps, epoch):
        dataset_mode_to_loss_dict = {
            'train': train_loss,
            'val': val_loss}
        if test_loss is not None:
            dataset_mode_to_loss_dict.update({'test': test_loss})

        # Construct loggable dict
        wandb_loss_dict = self.construct_loggable_dict(
            dataset_mode_to_loss_dict)

        if self.tradeoff_annealer is not None:
            wandb_loss_dict['tradeoff'] = self.tradeoff_annealer.curr_tradeoff

        wandb_loss_dict['step'] = steps
        wandb_loss_dict['epoch'] = epoch
        wandb_loss_dict['lr'] = self.optimizer.param_groups[0]['lr']
        wandb_loss_dict['checkpoint_time'] = (
            f'{time.time() - self.checkpoint_start:.3f}')
        self.checkpoint_start = time.time()

        # Log to wandb
        if self.gpu is None or self.gpu == 0:
            wandb.log(wandb_loss_dict, step=steps)

        # Log to stdout
        self.print_loss_dict(wandb_loss_dict)

        return wandb_loss_dict

    def summary_log(self, loss_dict, new_min):
        # No summary metrics written
        if self.c.mp_distributed:
            return 0

        # Do not update summary metrics if not min (min already updated)
        if not new_min:
            return 0

        loss_dict.update({'time': time.time() - self.train_start})
        # Always need to rewrite old summary loss dict, because wandb overrides
        # the summary dict when calling normal log
        lowest_dict = {f'best_{i}': j for i, j in loss_dict.items()}

        wandb.run.summary.update(lowest_dict)

    @staticmethod
    def safe_torch_to_float(val):
        if type(val) == torch.Tensor:
            return val.detach().cpu().numpy().item(0)
        else:
            return val

    @staticmethod
    def construct_loggable_dict(dataset_mode_to_loss_dict):
        wandb_loss_dict = dict()
        for dataset_mode, loss_dict in dataset_mode_to_loss_dict.items():
            for key, value in loss_dict.items():
                key = f'{dataset_mode}_{key}'
                if type(value) == dict:
                    for key2, value2 in value.items():
                        joint_key = f'{key}_{key2}'
                        wandb_loss_dict[joint_key] = (
                            Logger.safe_torch_to_float(value2))
                else:
                    wandb_loss_dict[key] = Logger.safe_torch_to_float(value)

        return wandb_loss_dict

    @staticmethod
    def print_loss_dict(loss_dict):
        train_keys = []
        val_keys = []
        test_keys = []
        summary_keys = []

        for key in loss_dict.keys():
            if 'train' in key:
                train_keys.append(key)
            elif 'val' in key:
                val_keys.append(key)
            elif 'test' in key:
                test_keys.append(key)
            else:
                summary_keys.append(key)

        line = ''
        for key in summary_keys:
            line += f'{key} {loss_dict[key]} | '
        line += f'\nTrain Stats\n'
        for key in train_keys:
            line += f'{key} {loss_dict[key]:.3f} | '
        line += f'\nVal Stats\n'
        for key in val_keys:
            line += f'{key} {loss_dict[key]:.3f} | '
        line += f'\nTest Stats\n'
        for key in test_keys:
            line += f'{key} {loss_dict[key]:.3f} | '
        line += '\n'
        print(line)

    def intermediate_log(self, loss_dict, num_steps, batch_index, epoch):
        """Log during mini-batches."""

        tb = 'train_batch'
        ld = loss_dict

        wandb_dict = dict(
            batch_index=batch_index,
            epoch=epoch)

        losses = dict()

        losses.update({
            f'{tb}_total_loss':
            ld['total_loss']})

        if tl := ld['label'].get('total_loss', False):
            losses.update({
                f'{tb}_label_total_loss': tl})

        if tl := ld['augmentation'].get('total_loss', False):
            losses.update({
                f'{tb}_augmentation_total_loss': tl})

        if val := ld['label'].get('cat_accuracy', False):
            losses.update({f'{tb}_label_accuracy': val})
        if val := ld['label'].get('num_mse_loss', False):
            losses.update({f'{tb}_label_num_mse': val})

        losses = {i: j.detach().cpu().item() for i, j in losses.items()}
        wandb_dict.update(losses)

        print(f'step: {num_steps}, {wandb_dict}')

        if self.gpu is None or self.gpu == 0:
            wandb.log(wandb_dict, step=num_steps)
