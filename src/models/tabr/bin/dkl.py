# Deep Kernel Learning

# >>>
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import delu
import gpytorch
import torch

import lib
from lib import KWArgs


# inspired by https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/KISSGP_Deep_Kernel_Regression_CUDA.html
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor, kernel):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel == 'sm':
            k = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=2)
        elif kernel == 'rbf':
            k = gpytorch.kernels.RBFKernel(ard_num_dims=2)
        else:
            raise ValueError('Unknown kernel')

        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(k),
            num_dims=2,
            grid_size=100,
        )
        self.feature_extractor = feature_extractor

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values 'nice'

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)  # type: ignore[code]


def concat_features(
    data: dict[str, dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    cat_cardinalities = (
        [len(x.unique()) for x in data['X_cat']['train'].T] if 'X_cat' in data else []
    )
    cat_encoder = lib.OneHotEncoder(cat_cardinalities) if cat_cardinalities else None

    X = {}
    for part in ['train', 'val', 'test']:
        x = [data['X_num'][part]]
        if 'X_bin' in data:
            x.append(data['X_bin'][part])
        if 'X_cat' in data:
            assert cat_encoder is not None
            x.append(cat_encoder(data['X_cat'][part]))
        X[part] = torch.cat(x, dim=1)
    return X


@dataclass
class Config:
    seed: int
    data: KWArgs  # lib.data.build_dataset
    mlp: KWArgs
    kernel: Literal['sm', 'rbf']
    lr: float
    weight_decay: float
    patience: Optional[int]
    n_epochs: Union[int, float]


def main(
    config: lib.JSONDict, output: Union[str, Path], *, force: bool = False
) -> Optional[lib.JSONDict]:
    warnings.filterwarnings('ignore')

    # >>> start
    if not lib.start(output, force=force):
        return None

    output = Path(output)
    report = lib.create_report(config)
    C = lib.make_config(Config, config)

    delu.random.seed(C.seed)
    device = lib.get_device()

    # >>> data
    dataset = (
        C.data if isinstance(C.data, lib.Dataset) else lib.build_dataset(**C.data)
    ).to_torch(device)
    assert dataset.is_regression
    dataset.data['Y'] = {k: v.float() for k, v in dataset.Y.items()}
    X = concat_features(dataset.data)

    # >>> model
    d_out = 2  # as in paper
    feature_extractor = lib.MLP(d_in=X['train'].shape[1], d_out=d_out, **C.mlp)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GPRegressionModel(
        X['train'],
        dataset.Y['train'],
        likelihood,
        feature_extractor,
        kernel=C.kernel,
    )
    report['prediction_type'] = None if dataset.is_regression else 'probs'
    model.to(device)

    # >>> training
    optimizer = torch.optim.AdamW(
        [
            {'params': model.feature_extractor.parameters()},
            {'params': model.covar_module.parameters()},
            {'params': model.mean_module.parameters()},
            {'params': model.likelihood.parameters()},  # type: ignore[code]
        ],
        lr=C.lr,
        weight_decay=C.weight_decay,
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    epoch = 0
    timer = delu.Timer()
    progress = delu.ProgressTracker(C.patience)
    training_log = []
    checkpoint_path = output / 'checkpoint.pt'

    @torch.inference_mode()
    def evaluate(parts: list[str]):
        model.eval()
        likelihood.eval()

        predictions = {}
        for part in parts:
            with torch.no_grad(), gpytorch.settings.use_toeplitz(
                False
            ), gpytorch.settings.fast_pred_var():
                preds = model(X[part]).mean

            all_preds = preds.cpu().numpy()
            if dataset.is_binclass:
                all_preds = all_preds[:, 1]
            predictions[part] = all_preds

        metrics = dataset.calculate_metrics(predictions, report['prediction_type'])

        return metrics, predictions

    def save_checkpoint():
        lib.dump_checkpoint(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'random_state': delu.random.get_state(),
                'progress': progress,
                'report': report,
                'timer': timer,
                'training_log': training_log,
            },
            output,
        )
        lib.dump_report(report, output)
        lib.backup_output(output)

    print()
    timer.run()
    while epoch < C.n_epochs:
        print(f'\n>>> {lib.try_get_relative_path(output)} | epoch {epoch}')

        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        out = model(X['train'])
        loss = -mll(out, dataset.Y['train'])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        metrics, predictions = evaluate(['val', 'test'])
        lib.print_metrics(mean_loss, metrics)
        training_log.append(
            {'mean_train_loss': epoch_loss, 'metrics': metrics, 'time': timer()}
        )

        progress.update(metrics['val']['score'])
        if progress.success:
            lib.celebrate()
            report['best_epoch'] = epoch
            report['metrics'] = metrics
            save_checkpoint()
            lib.dump_predictions(predictions, output)

        elif progress.fail or not lib.are_valid_predictions(predictions):
            break

        epoch += 1
        print()
    report['time'] = str(timer)

    with torch.inference_mode():
        model.load_state_dict(torch.load(checkpoint_path)['model'])
    report['metrics'], predictions = evaluate(['train', 'val', 'test'])
    lib.dump_predictions(predictions, output)
    lib.dump_summary(lib.summarize(report), output)
    save_checkpoint()
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_Function_cli(main)
