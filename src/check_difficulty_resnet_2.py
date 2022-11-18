import os
os.environ["PROJECT_DIR"] = "test"
from skorch_models import create_resnet_skorch
from skorch_models_regression import create_resnet_regressor_skorch
import openml
print("ho")
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import sys
sys.path.append("/storage/store/work/lgrinszt/ToyTabular")
print(sys.path)
from data.data_utils import *
import torch
from train import *
from sklearn.compose import TransformedTargetRegressor
import argparse
from sklearn.metrics import r2_score

def remove_high_cardinality(X, y, categorical_mask, threshold=20):
    high_cardinality_mask = X.nunique() > threshold
    print("high cardinality columns: {}".format(X.columns[high_cardinality_mask * categorical_mask]))
    n_high_cardinality = sum(categorical_mask * high_cardinality_mask)
    X = X.drop(X.columns[categorical_mask * high_cardinality_mask], axis=1)
    print("Removed {} high-cardinality categorical features".format(n_high_cardinality))
    categorical_mask = [categorical_mask[i] for i in range(len(categorical_mask)) if not (high_cardinality_mask[i] and categorical_mask[i])]
    return X, y, categorical_mask, n_high_cardinality



def remove_pseudo_categorical(X, y):
    """Remove columns where most values are the same"""
    pseudo_categorical_cols_mask = X.nunique() < 10
    print("Removed {} columns with pseudo-categorical values on {} columns".format(sum(pseudo_categorical_cols_mask),
                                                                                   X.shape[1]))
    X = X.drop(X.columns[pseudo_categorical_cols_mask], axis=1)
    return X, y, sum(pseudo_categorical_cols_mask)


def remove_rows_with_missing_values(X, y):
    missing_rows_mask = pd.isnull(X).any(axis=1)
    print("Removed {} rows with missing values on {} rows".format(sum(missing_rows_mask), X.shape[0]))
    X = X[~missing_rows_mask]
    y = y[~missing_rows_mask]
    return X, y


def remove_missing_values(X, y, threshold=0.7, return_missing_col_mask=True):
    """Remove columns where most values are missing, then remove any row with missing values"""
    missing_cols_mask = pd.isnull(X).mean(axis=0) > threshold
    print("Removed {} columns with missing values on {} columns".format(sum(missing_cols_mask), X.shape[1]))
    X = X.drop(X.columns[missing_cols_mask], axis=1)
    missing_rows_mask = pd.isnull(X).any(axis=1)
    print("Removed {} rows with missing values on {} rows".format(sum(missing_rows_mask), X.shape[0]))
    X = X[~missing_rows_mask]
    y = y[~missing_rows_mask]
    if return_missing_col_mask:
        return X, y, sum(missing_cols_mask), sum(missing_rows_mask), missing_cols_mask.values
    else:
        return X, y, sum(missing_cols_mask), sum(missing_rows_mask)


def balance(x, y):
    rng = np.random.RandomState(0)
    print("Balancing")
    print(X.shape)
    indices = [(y == i) for i in np.unique(y)]
    sorted_classes = np.argsort(
        list(map(sum, indices)))  # in case there are more than 2 classes, we take the two most numerous

    n_samples_min_class = sum(indices[sorted_classes[-2]])
    print("n_samples_min_class", n_samples_min_class)
    indices_max_class = rng.choice(np.where(indices[sorted_classes[-1]])[0], n_samples_min_class, replace=False)
    indices_min_class = np.where(indices[sorted_classes[-2]])[0]
    total_indices = np.concatenate((indices_max_class, indices_min_class))
    y = y[total_indices]
    indices_first_class = (y == sorted_classes[-1])
    indices_second_class = (y == sorted_classes[-2])
    y[indices_first_class] = 0
    y[indices_second_class] = 1

    return X.iloc[total_indices], y

# Create an argument parser
parser = argparse.ArgumentParser(description='Train a model on a dataset')

# Add the arguments
parser.add_argument('--device', type=str, default="cuda", help='Device to use')
parser.add_argument('--file', type=str, default="filename", help='Csv with all datasets')
parser.add_argument('--out_file', type=str, default="filename", help='filename to save')

# Parse the arguments
args = parser.parse_args()

df = pd.read_csv("data/aggregates/{}.csv".format(args.file))

regression = True

device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
print(device)

# resnet_config =  {"model_name": "rtdl_resnet",
#                   "model_type": "skorch",
#                   "model__use_checkpoints": True,
#                   "model__optimizer": "adamw",
#                   "model__lr_scheduler": True,
#                   "model__verbose":100,
#                   "model__batch_size": 512,
#                   "model__max_epochs": 300,
#                   "model__module__activation": "reglu",
#                   "model__module__normalization": "batchnorm",
#                   "model__module__n_layers": 8,
#                   "model__module__d": 256,
#                   "model__module__d_hidden_factor": 2,
#                   "model__module__hidden_dropout":  0.2,
#                   "model__module__residual_dropout":  0.2,
#                   "model__lr": 1e-3,
#                   "model__optimizer__weight_decay":  1e-7,
#                   "model__module__d_embedding": 128,
#                   "regression": False}

resnet_config =  {"model_name": "rtdl_resnet_regressor",
                  "model_type": "skorch",
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
                  "model__module__hidden_dropout":  0.2,
                  "model__module__residual_dropout":  0.2,
                  "model__lr": 1e-3,
                  "model__optimizer__weight_decay":  1e-7,
                  "model__module__d_embedding": 128,
                  "regression": True,
                  "data__regression": True,
                  "data__categorical": True,
                  "transformed_target": True}




res_df = pd.DataFrame()


for index, row in df.iterrows():
    #try:
    if not pd.isnull(row["dataset_id"]) and row["Remove"] != 1 and row["too_easy"] == 1 and row["Redundant"] != 1:
        prefix_to_skip = ["BNG", "RandomRBF", "GTSRB", "CovPokElec", "PCam"]
        if not(np.any([row["dataset_name"].startswith(prefix) for prefix in
                   prefix_to_skip]) or "mnist" in row["dataset_name"].lower() or "image" in row["dataset_name"].lower() or "cifar" in row["dataset_name"].lower() or row["dataset_id"] == 1414):
            print(row["dataset_name"])
            dataset_id = int(row["dataset_id"])
            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute
            )

            # try:
            #X = X.drop(X.columns[categorical_indicator], axis=1)
            num_categorical_columns = sum(categorical_indicator)
            original_n_samples, original_n_features = X.shape
            le = LabelEncoder()
            y = le.fit_transform(y)
            # with open("numerical_only/full/data_{}".format(name), "wb") as f:
            #    pickle.dump((x, y), f)
            binary_variables_mask = X.nunique() == 2
            for i in range(X.shape[1]):
                if binary_variables_mask[i]:
                    categorical_indicator[i] = True
            for i in range(X.shape[1]):
                if type(X.iloc[1, i]) == str:
                    categorical_indicator[i] = True
                    # X.iloc[:, i] = preprocessing.LabelEncoder().fit_transform(X.iloc[:, i])
            # for i in range(X.shape[1]):
            #    if categorical_indicator[i]:
            #        X.iloc[:, i] = preprocessing.LabelEncoder().fit_transform(X.iloc[:, i])

            pseudo_categorical_mask = X.nunique() < 10
            n_pseudo_categorical = 0
            cols_to_delete = []
            for i in range(X.shape[1]):
                if pseudo_categorical_mask[i]:
                    if not categorical_indicator[i]:
                        n_pseudo_categorical += 1
                        cols_to_delete.append(i)
            print("low card to delete")
            print(X.columns[cols_to_delete])
            print("{} low cardinality numerical removed".format(n_pseudo_categorical))
            X = X.drop(X.columns[cols_to_delete], axis=1)
            categorical_indicator = [categorical_indicator[i] for i in range(len(categorical_indicator)) if
                                     not i in cols_to_delete]
            X, y, categorical_indicator, num_high_cardinality = remove_high_cardinality(X, y, categorical_indicator, 20)
            print([X.columns[i] for i in range(X.shape[1]) if categorical_indicator[i]])
            X, y, num_columns_missing, num_rows_missing, missing_cols_mask = remove_missing_values(X, y, 0.2)
            categorical_indicator = [categorical_indicator[i] for i in range(len(categorical_indicator)) if
                                     not missing_cols_mask[i]]
            for i in range(X.shape[1]):
                if categorical_indicator[i]:
                    X.iloc[:, i] = LabelEncoder().fit_transform(X.iloc[:, i])
            #if X.shape[0] > 3000 and X.shape[1] > 3:
            X, y = balance(X, y)


            if "Transformation" in row.keys():
                if pd.isnull(row["Transformation"]):
                    transformation = "none"
                else:
                    transformation = row["Transformation"]
                y = transform_target(y,transformation)
            else:
                print("NO TRANSFORMATION")
            train_prop = 0.7
            train_prop = min(10000 / X.shape[0], train_prop)
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
                    ("num", numeric_transformer_sparse, [i for i in range(X.shape[1]) if not categorical_indicator[i]]),
                    ("cat", categorical_transformer, [i for i in range(X.shape[1]) if categorical_indicator[i]]),
                ]
            )

            processor_no_one_hot = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, [i for i in range(X.shape[1]) if not categorical_indicator[i]]),
                    ("cat", "passthrough", [i for i in range(X.shape[1]) if categorical_indicator[i]]),
                ]
            )
            processor_no_one_hot_sparse = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer_sparse, [i for i in range(X.shape[1]) if not categorical_indicator[i]]),
                    ("cat", "passthrough", [i for i in range(X.shape[1]) if categorical_indicator[i]]),
                ]
            )


            score_resnet_list = []
            score_linear_list = []
            if int((1 - train_prop) * X.shape[0]) > 10000:
                n_iters = 1
            elif int((1 - train_prop) * X.shape[0]) > 5000:
                n_iters = 3
            else:
                n_iters = 5
            print(X.shape)
            if X.shape[0] > 1000 and X.shape[1] > 3:
                for iter in range(n_iters):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_prop)
                    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
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
                        X_train_no_one_hot = processor_no_one_hot.fit_transform(X_train)
                        X_test_no_one_hot = processor_no_one_hot.transform(X_test)

                    except:
                        print("trying MaxAbsScaler")
                        X_train_one_hot = preprocessor_sparse.fit_transform(X_train)
                        X_test_one_hot = preprocessor_sparse.transform(X_test)
                        X_train_no_one_hot = processor_no_one_hot_sparse.fit_transform(X_train)
                        X_test_no_one_hot = processor_no_one_hot_sparse.transform(X_test)
                    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
                    X_train_one_hot, X_test_one_hot = np.array(X_train_one_hot), np.array(X_test_one_hot)
                    X_train_no_one_hot, X_test_no_one_hot = np.array(X_train_no_one_hot), np.array(X_test_no_one_hot)
                    X_train_one_hot, X_test_one_hot = X_train_one_hot.astype(np.float32), X_test_one_hot.astype(np.float32)
                    X_train_no_one_hot, X_test_no_one_hot = X_train_no_one_hot.astype(np.float32), X_test_no_one_hot.astype(np.float32)
                    if regression:
                        linear_model = TransformedTargetRegressor(regressor=LinearRegression(),
                                                                  transformer=QuantileTransformer(output_distribution="normal"))
                    else:
                        linear_model = LogisticRegression()

                    linear_model.fit(X_train_one_hot, y_train)
                    if regression:
                        score_linear = r2_score(y_test, linear_model.predict(X_test_one_hot))
                    else:
                        score_linear = linear_model.score(X_test_one_hot, y_test)  # accuracy
                    #if regression:
                    #    resnet = create_resnet_regressor_skorch(**resnet_config, id=dataset_id)
                    #else:
                    #    resnet = create_resnet_skorch(**resnet_config, id=dataset_id)
                    categorical_indicator = None
                    if resnet_config["regression"] == True:
                        y_train = y_train.reshape(-1, 1)
                        y_test = y_test.reshape(-1, 1)
                        y_train, y_test = y_train.astype(np.float32), y_test.astype(
                            np.float32)
                    else:
                        y_train = y_train.reshape(-1)
                        y_test = y_test.reshape(-1)
                    model, model_id = train_model(iter, X_train_no_one_hot, y_train, categorical_indicator, resnet_config)
                    train_score, val_score, score_resnet = evaluate_model(model, X_train_no_one_hot, y_train, None, None, X_test_no_one_hot,
                                                                        y_test, resnet_config, model_id, return_r2=True)
                    score_resnet = score_resnet #we want high = good so we take -RMSE
                    #if regression:
                    #    score_resnet = -mean_squared_error(y_test, resnet.predict(X_test), squared=False)  # rsme
                    #else:
                    #    score_resnet = resnet.score(X_test, y_test)
                    score_resnet_list.append(score_resnet)
                    score_linear_list.append(score_linear)
                    print("resnet score: ", score_resnet)
                    print("linear score: ", score_linear)
                print("Linear score: {}".format(score_linear_list))
                print("Resnet score: {}".format(score_resnet_list))
                if regression:
                    score_linear = np.nanmedian(score_linear_list)
                    score_resnet = np.nanmedian(score_resnet_list)
                else:
                    score_linear = np.mean(score_linear_list)
                    score_resnet = np.mean(score_resnet_list)
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
                    "heterogeneous": pd.NA}

                res_df = res_df.append(res_dic, ignore_index=True)
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
                "heterogeneous": pd.NA}

            res_df = res_df.append(res_dic, ignore_index=True)
    # except:
    #      print("FAILED")
    #      pass

res_df.to_csv("{}.csv".format(args.out_file))