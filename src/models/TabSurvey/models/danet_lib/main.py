from DAN_Task import DANetClassifier, DANetRegressor
import argparse
import os
import torch.distributed
import torch.backends.cudnn
from sklearn.metrics import accuracy_score, mean_squared_error
from data.dataset import get_data
from lib import normalize_reg_label
from qhoptim.pyt import QHAdam
from config.default import cfg
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch v1.4, DANet Task Training')
    parser.add_argument('-c', '--config', type=str, required=False, default='config/forest_cover_type.yaml', metavar="FILE", help='Path to config file')
    parser.add_argument('-g', '--gpu_id', type=str, default='1', help='GPU ID')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.backends.cudnn.benchmark = True if len(args.gpu_id) < 2 else False
    if args.config:
        cfg.merge_from_file(args.config)
    cfg.freeze()
    task = cfg.task
    seed = cfg.seed
    train_config = {'dataset': cfg.dataset, 'resume_dir': cfg.resume_dir, 'logname': cfg.logname}
    fit_config = dict(cfg.fit)
    model_config = dict(cfg.model)
    print('Using config: ', cfg)

    return train_config, fit_config, model_config, task, seed, len(args.gpu_id)

def set_task_model(task, std=None, seed=1):
    if task == 'classification':
        clf = DANetClassifier(
            optimizer_fn=QHAdam,
            optimizer_params=dict(lr=fit_config['lr'], weight_decay=1e-5, nus=(0.8, 1.0)),
            scheduler_params=dict(gamma=0.95, step_size=20),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            layer=model_config['layer'],
            base_outdim=model_config['base_outdim'],
            k=model_config['k'],
            drop_rate=model_config['drop_rate'],
            seed=seed
        )
        eval_metric = ['accuracy']

    elif task == 'regression':
        clf = DANetRegressor(
            std=std,
            optimizer_fn=QHAdam,
            optimizer_params=dict(lr=fit_config['lr'], weight_decay=fit_config['weight_decay'], nus=(0.8, 1.0)),
            scheduler_params=dict(gamma=0.95, step_size=fit_config['schedule_step']),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            layer=model_config['layer'],
            base_outdim=model_config['base_outdim'],
            k=model_config['k'],
            seed=seed
        )
        eval_metric = ['mse']
    return clf, eval_metric

if __name__ == '__main__':

    print('===> Setting configuration ...')
    train_config, fit_config, model_config, task, seed, n_gpu = get_args()
    logname = None if train_config['logname'] == '' else train_config['dataset'] + '/' + train_config['logname']
    print('===> Getting data ...')
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(train_config['dataset'])
    mu, std = None, None
    if task == 'regression':
        mu, std = y_train.mean(), y_train.std()
        print("mean = %.5f, std = %.5f" % (mu, std))
        y_train = normalize_reg_label(y_train, std, mu)
        y_valid = normalize_reg_label(y_valid, std, mu)
        y_test = normalize_reg_label(y_test, std, mu)

    clf, eval_metric = set_task_model(task, std, seed)

    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_valid, y_valid)],
        eval_name=['valid'],
        eval_metric=eval_metric,
        max_epochs=fit_config['max_epochs'], patience=fit_config['patience'],
        batch_size=fit_config['batch_size'], virtual_batch_size=fit_config['virtual_batch_size'],
        logname=logname,
        resume_dir=train_config['resume_dir'],
        n_gpu=n_gpu
    )

    preds_test = clf.predict(X_test)

    if task == 'classification':
        test_acc = accuracy_score(y_pred=preds_test, y_true=y_test)
        print(f"FINAL TEST ACCURACY FOR {train_config['dataset']} : {test_acc}")

    elif task == 'regression':
        test_mse = mean_squared_error(y_pred=preds_test, y_true=y_test)
        print(f"FINAL TEST MSE FOR {train_config['dataset']} : {test_mse}")
