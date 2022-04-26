import numpy as np
import pandas as pd
import wandb


def filter_col(col, df):
    return list(filter(lambda x: col in x, df.columns))


def get_df_from_project(project):
    """Load summary and config DataFrame for all runs of a project.

    Largely copied from wandb docs.
    """
    api = wandb.Api()
    runs = api.runs(f'anonymous_tab/{project}')

    summary_list = []
    config_list = []
    name_list = []

    for run in runs:
        # run.summary are the output key/values like accuracy. 
        # We call ._json_dict to omit large files.
        summary_list.append(run.summary._json_dict)
        # run.config is the input metrics.
        # We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items() if not k.startswith('_')})
        # run.name is the name of the run.
        name_list.append(run.name)

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({'name': name_list})
    df = pd.concat([name_df, config_df, summary_df], axis=1)

    return df


def get_rankings(df, val_metric, test_metrics, higher_is_better=False):
    """Assign ranking acc to val_metric to each model per split."""
    row_list = []
    for split, split_df in df.groupby('cv_index'):

        # get models and performances for that split
        models = split_df['exp_group']
        val_perfs = split_df[val_metric]
        test_perfs = split_df[test_metrics]

        # sort models, performances by val_perf
        sorting = np.argsort(val_perfs)
        if higher_is_better:
            sorting = sorting[::-1]
        models = models.values[sorting]
        val_perfs = val_perfs.values[sorting]
        test_perfs = test_perfs.values[sorting]

        # for each model report ranking/performance in each split
        for ranking, model in enumerate(models):
            row_list.append([
                model, split, ranking,
                val_perfs[ranking], *test_perfs[ranking]])

    rankings_df = pd.DataFrame(
        data=row_list,
        columns=[
            'exp_group', 'cv_index', 'ranking', val_metric, *test_metrics])

    # add rmse
    for test_metric in test_metrics:
        if 'mse' in test_metric:
            rankings_df[test_metric.replace('mse', 'rmse')] = np.sqrt(
                rankings_df[test_metric])

    return rankings_df


def report_losses(rankings_df):
    """Select model in each split by ranking."""
    losses = rankings_df[rankings_df.ranking == 0]
    for metric in rankings_df.columns:
        if metric in ['exp_group', 'cv_index', 'ranking']:
            continue
        loss = losses[metric]
        mean, std = loss.mean(), loss.std()
        if 'accuracy' in metric:
            mean *= 100
            std *= 100
            print(f'Metric {metric}: {mean:.2f} \\pm {std:.2f}')

        elif 'cat' in metric:
            print(f'Metric {metric}: {mean:.3f} \\pm {std:.3f}')

        else:
            print(f'Metric {metric}: {mean:.4f} \\pm {std:.4f}')

        if 'rmse_loss_unstd' in metric:
            std /= np.sqrt(len(loss))
            print(f'Metric {metric}: {mean:.4f} \\pm {std:.4f} (std_err)')

    return losses
