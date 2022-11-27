import os

os.environ["PROJECT_DIR"] = "test"
import openml

import sys

sys.path.append(".")
print(sys.path)
import torch
import pandas as pd
from train import *
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
# import train test split
from sklearn.model_selection import train_test_split
import argparse
import traceback
from preprocessing.preprocessing import preprocessing
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

# Create an argument parser
parser = argparse.ArgumentParser(description='Train a model on a dataset')

# Add the arguments
parser.add_argument('--device', type=str, default="cuda", help='Device to use')
parser.add_argument('--file', type=str, default="filename", help='Csv with all datasets')
parser.add_argument('--datasets', nargs='+', type=int, default=[], help='Datasets to use')
parser.add_argument('--out_file', type=str, default="filename", help='filename to save')
parser.add_argument('--regression', action='store_true', help='True if regression, false otherwise')
parser.add_argument('--categorical', action='store_true')
parser.add_argument('--all', action='store_true', help="Whether to check all datasets or only those already "
                                                       "deemed too easy with a HGBT")
parser.add_argument('--remove_model', nargs='+', help='List of models not to try')
# Parse the arguments
args = parser.parse_args()


device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() and not args.device == "cpu" else 'cpu'
print(device)

if args.remove_model is None:
    args.remove_model = []

resnet_config = {"model_type": "skorch",
                 "model__use_checkpoints": True,
                 "model__optimizer": "adamw",
                 "model__lr_scheduler": True,
                 "model__batch_size": 512,
                 "model__max_epochs": 300,
                 "model__module__activation": "reglu",
                 "model__module__normalization": "batchnorm",
                 "model__module__n_layers": 8,
                 "model__module__d": 256,
                 "model__module__d_hidden_factor": 2,
                 "model__module__hidden_dropout": 0.2,
                 "model__module__residual_dropout": 0.2,
                 "model__lr": 1e-3,
                 "model__optimizer__weight_decay": 1e-7,
                 "model__module__d_embedding": 128,
                 "model__verbose": 0,
                 "model__device": device}

if args.regression:
    resnet_config["model_name"] = "rtdl_resnet_regressor"
    resnet_config["regression"] = True
    resnet_config["data__regression"] = True
    resnet_config["transformed_target"] = True
else:
    resnet_config["model_name"] = "rtdl_resnet"
    resnet_config["regression"] = False
    resnet_config["data__regression"] = False
    resnet_config["transformed_target"] = False

if args.categorical:
    resnet_config["data__categorical"] = True
else:
    resnet_config["data__categorical"] = False

#TODO make a proper function
if len(args.datasets) == 0:
    df = pd.read_csv("../data/aggregates/{}.csv".format(args.file))
else:
    # Create a dataframe with columns dataset_name dataset_id and Remove and too_easy and Redundant
    df = pd.DataFrame(columns=["dataset_name", "dataset_id", "Remove"])
    df["dataset_id"] = args.datasets
    df["dataset_name"] = "default"
    df["Remove"] = 0
    df["too_easy"] = 1
    df["Redundant"] = 0


res_df = pd.DataFrame()



for index, row in df.iterrows():
    try:
        if not pd.isnull(row["dataset_id"]) and row["Remove"] != 1 and (row["too_easy"] == 1 or args.all) and row["Redundant"] != 1:
            prefix_to_skip = ["BNG", "RandomRBF", "GTSRB", "CovPokElec", "PCam"]
            if not (np.any([row["dataset_name"].startswith(prefix) for prefix in
                            prefix_to_skip]) or "mnist" in row["dataset_name"].lower() or "image" in row[
                        "dataset_name"].lower() or "cifar" in row["dataset_name"].lower() or row["dataset_id"] == 1414):
                print(row["dataset_name"])
                print("Downloading dataset")
                dataset_id = int(row["dataset_id"])
                dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
                print("Downloading data")
                X, y, categorical_indicator, attribute_names = dataset.get_data(
                    dataset_format="dataframe", target=dataset.default_target_attribute
                )
                print("Done")

                try:
                    transformation = None
                    if "Transformation" in row.keys():
                        if not pd.isnull(row["Transformation"]):
                            transformation = row["Transformation"]

                    X, y, categorical_indicator, num_high_cardinality, num_columns_missing, num_rows_missing, \
                    num_categorical_columns, n_pseudo_categorical, original_n_samples, original_n_features = \
                        preprocessing(X, y, categorical_indicator, categorical=args.categorical,
                                      regression=args.regression, transformation=transformation)
                    if X.shape[1] > 2000:
                        res_dic = {
                            "dataset_id": dataset_id,
                            "dataset_name": dataset.name,
                            "original_n_samples": original_n_samples,
                            "original_n_features": original_n_features,
                            "num_categorical_columns": num_categorical_columns,
                            "num_pseudo_categorical_columns": n_pseudo_categorical,
                            "num_columns_missing": num_columns_missing,
                            "num_rows_missing": num_rows_missing,
                            "too_easy": pd.NA,
                            "score_resnet": pd.NA,
                            "score_linear": pd.NA,
                            "score_hgbt": pd.NA,
                            "score_tree": pd.NA,
                            "heterogeneous": pd.NA,
                            "n_samples": X.shape[0]}
                        res_df = res_df.append(res_dic, ignore_index=True)
                        res_df.to_csv("{}.csv".format(args.out_file))
                        continue
                    train_prop = 0.7
                    train_prop = min(15000 / X.shape[0], train_prop)
                    numeric_transformer = StandardScaler()
                    numeric_transformer_sparse = MaxAbsScaler()
                    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ("num", numeric_transformer, [i for i in range(X.shape[1]) if not categorical_indicator[i]]),
                            ("cat", categorical_transformer, [i for i in range(X.shape[1]) if categorical_indicator[i]]),
                        ]
                    )
                    preprocessor_sparse = ColumnTransformer(
                        transformers=[
                            ("num", numeric_transformer_sparse,
                             [i for i in range(X.shape[1]) if not categorical_indicator[i]]),
                            ("cat", categorical_transformer, [i for i in range(X.shape[1]) if categorical_indicator[i]]),
                        ]
                    )

                    score_resnet_list = []
                    score_linear_list = []
                    score_hgbt_list = []
                    score_tree_list = []
                    if args.regression:
                        # same for r2
                        resnet_r2_list = []
                        linear_r2_list = []
                        hgbt_r2_list = []
                        tree_r2_list = []
                    if int((1 - train_prop) * X.shape[0]) > 10000:
                        n_iters = 1
                    elif int((1 - train_prop) * X.shape[0]) > 5000:
                        n_iters = 3
                    else:
                        n_iters = 5
                    print(X.shape)
                    if X.shape[0] > 3000 and X.shape[1] > 3:
                        score_resnet, score_linear, score_hgbt, score_tree = np.nan, np.nan, np.nan, np.nan
                        for iter in range(n_iters):
                            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop,
                                                                                random_state=np.random.RandomState(iter))
                            X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(
                                y_train), np.array(y_test)
                            if resnet_config["regression"] == True:
                                y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
                                y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)
                            if X_test.shape[0] > 30000:  # for speed
                                indices = np.random.choice(X_test.shape[0], 30000, replace=False)
                                try:
                                    X_test = X_test.iloc[indices]
                                except:
                                    X_test = X_test[indices]
                                y_test = y_test[indices]
                            try:
                                X_train_one_hot = preprocessor.fit_transform(X_train)
                                X_test_one_hot = preprocessor.transform(X_test)
                                X_train_no_one_hot = np.zeros_like(X_train)
                                X_test_no_one_hot = np.zeros_like(X_test)
                                # not column transformer to preserve order
                                for i in range(X_train.shape[1]):
                                    if categorical_indicator[i]:
                                        X_train_no_one_hot[:, i] = X_train[:, i]
                                        X_test_no_one_hot[:, i] = X_test[:, i]
                                    else:
                                        X_train_no_one_hot[:, i] = numeric_transformer.fit_transform(
                                            X_train[:, i].reshape(-1, 1)).reshape(-1)
                                        X_test_no_one_hot[:, i] = numeric_transformer.transform(
                                            X_test[:, i].reshape(-1, 1)).reshape(-1)

                            except:
                                print("trying MaxAbsScaler")
                                X_train_one_hot = preprocessor_sparse.fit_transform(X_train)
                                X_test_one_hot = preprocessor_sparse.transform(X_test)
                                X_train_no_one_hot = np.zeros_like(X_train)
                                X_test_no_one_hot = np.zeros_like(X_test)
                                for i in range(X_train.shape[1]):
                                    if categorical_indicator[i]:
                                        X_train_no_one_hot[:, i] = X_train[:, i]
                                        X_test_no_one_hot[:, i] = X_test[:, i]
                                    else:
                                        X_train_no_one_hot[:, i] = numeric_transformer_sparse.fit_transform(
                                            X_train[:, i].reshape(-1, 1)).reshape(-1)
                                        X_test_no_one_hot[:, i] = numeric_transformer_sparse.transform(
                                            X_test[:, i].reshape(-1, 1)).reshape(-1)

                            y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
                            X_train_one_hot, X_test_one_hot = np.array(X_train_one_hot), np.array(X_test_one_hot)
                            X_train_no_one_hot, X_test_no_one_hot = np.array(X_train_no_one_hot), np.array(
                                X_test_no_one_hot)
                            X_train_one_hot, X_test_one_hot = X_train_one_hot.astype(np.float32), X_test_one_hot.astype(
                                np.float32)
                            X_train_no_one_hot, X_test_no_one_hot = X_train_no_one_hot.astype(
                                np.float32), X_test_no_one_hot.astype(np.float32)

                            if args.regression:
                                if not "linear" in args.remove_model:
                                    linear_model = TransformedTargetRegressor(regressor=LinearRegression(),
                                                                          transformer=QuantileTransformer(
                                                                              output_distribution="normal"))
                                if not "hgbt" in args.remove_model:
                                    hgbt = HistGradientBoostingRegressor(categorical_features=categorical_indicator)
                                if not "tree" in args.remove_model:
                                    tree = DecisionTreeRegressor()
                            else:
                                if not "linear" in args.remove_model:
                                    linear_model = LogisticRegression()
                                if not "hgbt" in args.remove_model:
                                    hgbt = HistGradientBoostingClassifier(categorical_features=categorical_indicator)
                                if not "tree" in args.remove_model:
                                    tree = DecisionTreeClassifier()

                            if not "linear" in args.remove_model:
                                linear_model.fit(X_train_one_hot, y_train)
                            if not "hgbt" in args.remove_model:
                                hgbt.fit(X_train_no_one_hot, y_train)
                            if not "tree" in args.remove_model:
                                tree.fit(X_train_one_hot, y_train)
                            if args.regression:
                                if not "linear" in args.remove_model:
                                    score_linear = -mean_squared_error(y_test, linear_model.predict(X_test_one_hot),
                                                                       squared=False)
                                    r2_linear = r2_score(y_test, linear_model.predict(X_test_one_hot))
                                print("Linear model score: ", score_linear)
                                print("Linear model r2: ", r2_linear)
                                if not "hgbt" in args.remove_model:
                                    score_hgbt = -mean_squared_error(y_test, hgbt.predict(
                                        X_test_no_one_hot), squared=False)
                                    r2_hgbt = r2_score(y_test, hgbt.predict(X_test_no_one_hot))
                                    print("HGBT score: ", score_hgbt)
                                    print("HGBT r2: ", r2_hgbt)

                                if not "tree" in args.remove_model:
                                    score_tree = -mean_squared_error(y_test, tree.predict(
                                        X_test_one_hot), squared=False)
                                    r2_tree = r2_score(y_test, tree.predict(X_test_one_hot))
                                    print("Tree score: ", score_tree)
                                    print("Tree r2: ", r2_tree)
                            else:
                                if not "linear" in args.remove_model:
                                    score_linear = linear_model.score(X_test_one_hot, y_test)  # accuracy
                                    print("Linear model score: ", score_linear)
                                if not "hgbt" in args.remove_model:
                                    score_hgbt = hgbt.fit(X_train_no_one_hot, y_train).score(X_test_no_one_hot, y_test)
                                    print("HGBT score: ", score_hgbt)
                                if not "tree" in args.remove_model:
                                    score_tree = tree.fit(X_train_one_hot, y_train).score(X_test_one_hot, y_test)
                                    print("Tree score: ", score_tree)

                            if not "resnet" in args.remove_model:
                                if resnet_config["regression"] == True:
                                    y_train = y_train.reshape(-1, 1)
                                    y_test = y_test.reshape(-1, 1)
                                    y_train, y_test = y_train.astype(np.float32), y_test.astype(
                                        np.float32)
                                else:
                                    y_train = y_train.reshape(-1)
                                    y_test = y_test.reshape(-1)
                                    print("Number of classes: ", len(np.unique(y_train)))
                                    print("Number of classes max: ", np.max(y_train))
                                # Give the true number of categories to the model
                                categories = []
                                for i in range(len(categorical_indicator)):
                                    if categorical_indicator[i]:
                                        categories.append(int(np.max(X.iloc[:, i]) + 1))
                                resnet_config["model__categories"] = categories
                                model, model_id = train_model(iter, X_train_no_one_hot, y_train, categorical_indicator if len(categorical_indicator) > 0 else None,
                                                              resnet_config)
                                train_score, val_score, score_resnet = evaluate_model(model, X_train_no_one_hot, y_train, None,
                                                                                      None, X_test_no_one_hot,
                                                                                      y_test, resnet_config, model_id,
                                                                                      return_r2=False)
                                if args.regression:
                                    _, _, r2_resnet = evaluate_model(model, X_train_no_one_hot, y_train, None,
                                                                        None, X_test_no_one_hot,
                                                                        y_test, resnet_config, model_id,
                                                                        return_r2=True)
                                    print("Resnet r2: ", r2_resnet)
                                print("Resnet score: ", score_resnet)
                                print("Resnet train score: ", train_score)
                                print("Resnet val score: ", val_score)

                                if args.regression:
                                    score_resnet = -score_resnet  # we want high = good so we take -RMSE
                            else:
                                score_resnet = np.nan
                                r2_resnet = np.nan
                            score_resnet_list.append(score_resnet)
                            score_linear_list.append(score_linear)
                            score_hgbt_list.append(score_hgbt)
                            score_tree_list.append(score_tree)
                            print("resnet score: ", score_resnet)
                            print("linear score: ", score_linear)
                            print("hgbt score: ", score_hgbt)
                            print("tree score: ", score_tree)
                            if args.regression:
                                resnet_r2_list.append(r2_resnet)
                                linear_r2_list.append(r2_linear)
                                hgbt_r2_list.append(r2_hgbt)
                                tree_r2_list.append(r2_tree)
                        print("Linear score: {}".format(score_linear_list))
                        print("Resnet score: {}".format(score_resnet_list))
                        print("HGBT score: {}".format(score_hgbt_list))
                        print("Tree score: {}".format(score_tree_list))
                        if args.regression:
                            print("Linear r2: {}".format(linear_r2_list))
                            print("Resnet r2: {}".format(resnet_r2_list))
                            print("HGBT r2: {}".format(hgbt_r2_list))
                            print("Tree r2: {}".format(tree_r2_list))
                            score_linear = np.nanmedian(score_linear_list)
                            score_resnet = np.nanmedian(score_resnet_list)
                            score_hgbt = np.nanmedian(score_hgbt_list)
                            score_tree = np.nanmedian(score_tree_list)
                            r2_linear = np.nanmedian(linear_r2_list)
                            r2_resnet = np.nanmedian(resnet_r2_list)
                            r2_hgbt = np.nanmedian(hgbt_r2_list)
                            r2_tree = np.nanmedian(tree_r2_list)
                        else:
                            score_linear = np.mean(score_linear_list)
                            score_resnet = np.mean(score_resnet_list)
                            score_hgbt = np.mean(score_hgbt_list)
                            score_tree = np.mean(score_tree_list)
                            r2_linear = np.nan
                            r2_resnet = np.nan
                            r2_hgbt = np.nan
                            r2_tree = np.nan
                        if args.regression:
                            if (score_resnet - score_linear) < -0.05 * score_resnet:
                                too_easy = True
                            else:
                                too_easy = False
                        else:
                            if (score_resnet - score_linear) < 0.05 * score_resnet:
                                too_easy = True
                            else:
                                too_easy = False

                        res_dic = {
                            "dataset_id": dataset_id,
                            "dataset_name": dataset.name,
                            "original_n_samples": original_n_samples,
                            "original_n_features": original_n_features,
                            "num_categorical_columns": num_categorical_columns,
                            "num_pseudo_categorical_columns": n_pseudo_categorical,
                            "num_columns_missing": num_columns_missing,
                            "num_rows_missing": num_rows_missing,
                            "too_easy": too_easy,
                            "score_resnet": score_resnet,
                            "score_linear": score_linear,
                            "score_hgbt": score_hgbt,
                            "score_tree": score_tree,
                            "r2_resnet": r2_resnet,
                            "r2_linear": r2_linear,
                            "r2_hgbt": r2_hgbt,
                            "r2_tree": r2_tree,
                            "heterogeneous": pd.NA,
                            "n_samples": X.shape[0],
                            "too_small": False}

                        res_df = res_df.append(res_dic, ignore_index=True)
                        res_df.to_csv("{}.csv".format(args.out_file))
                    else:
                        print("dataset too small after preprocessing")
                        res_dic = {
                            "dataset_id": dataset_id,
                            "dataset_name": dataset.name,
                            "original_n_samples": original_n_samples,
                            "original_n_features": original_n_features,
                            "num_categorical_columns": num_categorical_columns,
                            "num_pseudo_categorical_columns": n_pseudo_categorical,
                            "num_columns_missing": num_columns_missing,
                            "num_rows_missing": num_rows_missing,
                            "too_easy": pd.NA,
                            "score_resnet": pd.NA,
                            "score_linear": pd.NA,
                            "score_hgbt": pd.NA,
                            "score_tree": pd.NA,
                            "heterogeneous": pd.NA,
                            "n_samples": X.shape[0],
                            "too_small": True}
                        res_df = res_df.append(res_dic, ignore_index=True)
                        res_df.to_csv("{}.csv".format(args.out_file))

                except:
                    print("FAILED")
                    print(traceback.format_exc())
                    pass
            else:
                res_dic = {
                    "dataset_id": row["dataset_id"],
                    "dataset_name": row["dataset_name"],
                    "original_n_samples": pd.NA,
                    "original_n_features": pd.NA,
                    "num_categorical_columns": pd.NA,
                    "num_pseudo_categorical_columns": pd.NA,
                    "num_columns_missing": pd.NA,
                    "num_rows_missing": pd.NA,
                    "too_easy": pd.NA,
                    "score_resnet": pd.NA,
                    "score_linear": pd.NA,
                    "score_hgbt": pd.NA,
                    "score_tree": pd.NA,
                    "heterogeneous": pd.NA,
                    "n_samples": pd.NA,
                    "too_small": pd.NA}

                res_df = res_df.append(res_dic, ignore_index=True)
    except:
        print("FAILED on dataset download")
        print(traceback.format_exc())
        res_dic = {
            "dataset_id": row["dataset_id"],
            "dataset_name": row["dataset_name"],
            "original_n_samples": pd.NA,
            "original_n_features": pd.NA,
            "num_categorical_columns": pd.NA,
            "num_pseudo_categorical_columns": pd.NA,
            "num_columns_missing": pd.NA,
            "num_rows_missing": pd.NA,
            "too_easy": pd.NA,
            "score_resnet": pd.NA,
            "score_linear": pd.NA,
            "score_hgbt": pd.NA,
            "score_tree": pd.NA,
            "heterogeneous": pd.NA,
            "n_samples": pd.NA,
            "too_small": pd.NA}

        res_df = res_df.append(res_dic, ignore_index=True)

res_df.to_csv("{}.csv".format(args.out_file))
