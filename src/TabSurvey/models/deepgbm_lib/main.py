import numpy as np

from models.deepgbm_lib import AdamW
from models.deepgbm_lib.models.DeepGBM import DeepGBM
from models.deepgbm_lib.models.EmbeddingModel import EmbeddingModel
from models.deepgbm_lib.preprocess.preprocessing_cat import CatEncoder
from models.deepgbm_lib.preprocess.preprocessing_num import NumEncoder
from models.deepgbm_lib.utils.gbdt import TrainGBDT, SubGBDTLeaf_cls, get_infos_from_gbms
from models.deepgbm_lib.utils.helper import outputFromEmbeddingModel, eval_metrics, printMetric

from models.deepgbm_lib.trainModel import trainModel, evaluateModel, makePredictions

import models.deepgbm_lib.config as config

'''
    Preprocess the data
    
        for GBDT2NN: Fill NaNs, Ordinal encode and then binary encode categoric features, Scale numeric features
        for CatNN: Fill NaNs, Bucketize numeric features, Filter categoric features, Ordinal encode all features
    
    Parameters:
        - df: dataframe with all trainings data
        - num_col: list of all numerical columns
        - cat_col: list of all categorical columns
        - label_col: name of label column
        
    Returns:
        - train_num: data and label preprocessed for training
        - test_num: data and label preprocessed for testing
        - train_cat: data preprocessed for training CatNN
        - test_cat: data preprocessed for testing CatNN
        - feature sizes: list containing the number of different features in every column
'''


def preprocess(df, num_col, cat_col, label_col):
    print('\n################### Preprocess data ##################################\n')

    # Split the data into train and test data
    train_df = df.sample(frac=config.config['rate'])
    test_df = df.drop(train_df.index)

    train_df = train_df.reset_index()
    test_df = test_df.reset_index()

    # Preprocess data for CatNN
    ce = CatEncoder(cat_col, num_col)
    train_x_cat, feature_sizes = ce.fit_transform(train_df.copy())
    test_x_cat = ce.transform(test_df.copy()).astype('int32')

    # Preprocess data for GBDT2NN
    ne = NumEncoder(cat_col, num_col, label_col)
    train_x, train_y = ne.fit_transform(train_df.copy())
    test_x, test_y = ne.transform(test_df.copy())

    return (train_x, train_y), (test_x, test_y), train_x_cat, test_x_cat, feature_sizes, ce, ne


'''

    Train the model
    
        1. Train a GBDT and distill the knowledge from its leafs
        2. Train a Embedding Model using the leaf predictions as data
        3. Train a DeepGBM model (consisting of a GBDT2NN and a CatNN) using all data and the outputs from the Embedding Model
    
    Parameters:
        - train_num: data and label preprocessed for training
        - test_num: data and label preprocessed for testing
        - train_cat: data preprocessed for training CatNN
        - test_cat: data preprocessed for testing CatNN
        - feature sizes: list containing the number of different features in every col
        
    Returns:
        - deepgbm_model: trained model
        - optimizer: optimizer used during training
'''


def train(train_num, test_num, train_cat, test_cat, feature_sizes, save_model=False):
    x_train, y_train = train_num
    x_test, y_test = test_num

    print('\n################### Train model #######################################\n')

    print("\nTrain GBDT and distill knowledge from it...")
    # Train GBDT and distill knowledge from it
    gbm = TrainGBDT(x_train, y_train, x_test, y_test)
    gbms = SubGBDTLeaf_cls(x_train, x_test, gbm)
    used_features, tree_outputs, leaf_preds, test_leaf_preds, group_average, max_ntree_per_split = get_infos_from_gbms(
        gbms, min_len_features=x_train.shape[1])
    n_models = len(used_features)

    # Train embedding model
    print("\nTrain embedding model...")
    emb_model = EmbeddingModel(n_models, max_ntree_per_split, n_output=y_train.shape[1])

    optimizer = AdamW(emb_model.parameters(), lr=config.config['emb_lr'], weight_decay=config.config['l2_reg'])
    tree_outputs = np.asarray(tree_outputs).reshape((n_models, leaf_preds.shape[0])).transpose((1, 0))

    trainModel(emb_model, leaf_preds, y_train, tree_outputs, test_leaf_preds, y_test,
               optimizer, epochs=config.config['emb_epochs'])
    output_w, output_b, tree_outputs = outputFromEmbeddingModel(emb_model, leaf_preds, y_train.shape[1], n_models)

    # Train DeepGBM model
    print("\nTrain DeepGBM model...")
    deepgbm_model = DeepGBM(nume_input_size=x_train.shape[1],
                            used_features=np.asarray(used_features, dtype=np.int64),
                            output_w=output_w,
                            output_b=output_b,
                            cate_field_size=train_cat.shape[1],
                            feature_sizes=feature_sizes)

    optimizer = AdamW(deepgbm_model.parameters(), lr=config.config['lr'], weight_decay=config.config['l2_reg'],
                      amsgrad=False, model_decay_opt=deepgbm_model, weight_decay_opt=config.config['l2_reg_opt'],
                      key_opt='deepgbm')

    x_train = np.concatenate([x_train, np.zeros((x_train.shape[0], 1), dtype=np.float32)], axis=-1)
    x_test = np.concatenate([x_test, np.zeros((x_test.shape[0], 1), dtype=np.float32)], axis=-1)

    _, _, loss_history, val_loss_history = trainModel(deepgbm_model, x_train, y_train, tree_outputs, x_test, y_test,
                                                      optimizer,
                                                      train_x_cat=train_cat, test_x_cat=test_cat,
                                                      epochs=config.config['epochs'],
                                                      early_stopping_rounds=config.config['early_stopping'],
                                                      save_model=save_model)

    return deepgbm_model, optimizer, loss_history, val_loss_history


'''

    Evaluate the model (metric depends on given task)
    
    Parameters:
        - model: trained model
        - test_num: data and label preprocessed for testing
        - test_cat: data preprocessed for testing CatNN
        
    Returns:
        - test_loss: loss computed on the testing data
        - preds: predicitions on test data
'''


def evaluate(model, test_num, test_cat):
    x_test, y_test = test_num
    x_test = np.concatenate([x_test, np.zeros((x_test.shape[0], 1), dtype=np.float32)], axis=-1)

    print('\n################### Evaluate model #######################################\n')

    test_loss, preds = evaluateModel(model, x_test, y_test, test_cat)
    metric = eval_metrics(config.config['task'], y_test, preds)
    printMetric(config.config['task'], metric, test_loss)

    return test_loss, preds


'''

    Make predicitions on new data
    
    Parameters:
        - model: trained model
        - data: new unlabeled test data
        - ce: CatEncoder from the preprocessing
        - ne: NumEncoder from thre preprocessing
        
    Returns:
        - predicitions on data
'''


def predict(model, data, ce, ne):
    # Transform data properly
    data_cat = ce.transform(data.copy()).astype('int32')
    data_num = ne.transform(data.copy())

    data_num = np.concatenate([data_num, np.zeros((data_num.shape[0], 1), dtype=np.float32)], axis=-1)

    print('\n################### Make predictions #######################################\n')

    return makePredictions(model, data_num, data_cat)


##########################################################################################################################
import argparse
import pandas as pd

num_col = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
           'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
           'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
cat_col = ['SEX', 'EDUCATION', 'MARRIAGE']
label_col = 'default.payment.next.month'


def main():
    parser = argparse.ArgumentParser(description='DeepGBM Models')
    parser.add_argument('-data', type=str)
    args = parser.parse_args()

    if args.data is None:
        print("Specify file location with -data <filename>")
        return

    df = pd.read_csv(args.data)

    train_num, test_num, train_cat, test_cat, feature_sizes, _, _ = preprocess(df, num_col, cat_col, label_col)
    model, _ = train(train_num, test_num, train_cat, test_cat, feature_sizes)
    test_loss, preds = evaluate(model, test_num, test_cat)


if __name__ == '__main__':
    main()
