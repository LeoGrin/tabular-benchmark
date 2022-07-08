from DAN_Task import DANetClassifier, DANetRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from lib.multiclass_utils import infer_output_dim
from lib.utils import normalize_reg_label
import numpy as np
import argparse
from data.dataset import get_data
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch v1.4, DANet Testing')
    parser.add_argument('-d', '--dataset', type=str, default='forest', help='Dataset Name for extracting data')
    parser.add_argument('-m', '--model_file', type=str, default='./weights/forest_layer32.pth', metavar="FILE", help='Inference model path')
    parser.add_argument('-g', '--gpu_id', type=str, default='1', help='GPU ID')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    dataset = args.dataset
    model_file = args.model_file
    task = 'regression' if dataset in ['year', 'yahoo', 'MSLR'] else 'classification'

    return dataset, model_file, task, len(args.gpu_id)

def set_task_model(task):
    if task == 'classification':
        clf = DANetClassifier()
        metric = accuracy_score
    elif task == 'regression':
        clf = DANetRegressor()
        metric = mean_squared_error
    return clf, metric

def prepare_data(task, y_train, y_valid, y_test):
    output_dim = 1
    mu, std = None, None
    if task == 'classification':
        output_dim, train_labels = infer_output_dim(y_train)
        target_mapper = {class_label: index for index, class_label in enumerate(train_labels)}
        y_train = np.vectorize(target_mapper.get)(y_train)
        y_valid = np.vectorize(target_mapper.get)(y_valid)
        y_test = np.vectorize(target_mapper.get)(y_test)

    elif task == 'regression':
        mu, std = y_train.mean(), y_train.std()
        print("mean = %.5f, std = %.5f" % (mu, std))
        y_train = normalize_reg_label(y_train, mu, std)
        y_valid = normalize_reg_label(y_valid, mu, std)
        y_test = normalize_reg_label(y_test, mu, std)

    return output_dim, std, y_train, y_valid, y_test

if __name__ == '__main__':
    dataset, model_file, task, n_gpu = get_args()
    print('===> Getting data ...')
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(dataset)
    output_dim, std, y_train, y_valid, y_test = prepare_data(task, y_train, y_valid, y_test)
    clf, metric = set_task_model(task)

    filepath = model_file
    clf.load_model(filepath, input_dim=X_test.shape[1], output_dim=output_dim, n_gpu=n_gpu)

    preds_test = clf.predict(X_test)
    test_value = metric(y_pred=preds_test, y_true=y_test)

    if task == 'classification':
        print(f"FINAL TEST ACCURACY FOR {dataset} : {test_value}")

    elif task == 'regression':
        print(f"FINAL TEST MSE FOR {dataset} : {test_value}")
