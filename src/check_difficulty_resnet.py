import os
os.environ["PROJECT_DIR"] = "test"
from skorch_models import create_resnet_skorch
from skorch_models_regression import create_resnet_regressor_skorch
import openml
print("ho")
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import sys
sys.path.append("/Users/leo/PycharmProjets/ToyTabular")
print(sys.path)
from data.data_utils import *
import torch
from train import *
from sklearn.compose import TransformedTargetRegressor

df = pd.read_csv("/Users/leo/PycharmProjets/ToyTabular/data/aggregates/all_datasets_regression_numerical.csv")

regression = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
                  "model__max_epochs": 100,
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
                  "transformed_target":True}




res_df = pd.DataFrame()


for index, row in df.iterrows():
    #try:
    if not pd.isnull(row["dataset_id"]) and row["Remove"] != 1 and row["too_easy"] == 1 and row["Redundant"] != 1:
        prefix_to_skip = ["BNG", "RandomRBF", "GTSRB", "CovPokElec", "PCam"]
        if not(np.any([row["dataset_name"].startswith(prefix) for prefix in
                   prefix_to_skip]) or "mnist" in row["dataset_name"].lower() or "image" in row["dataset_name"].lower() or "cifar" in row["dataset_name"].lower() or row["dataset_id"] == 1414 or row["dataset_id"] == 40753):
            print(row["dataset_name"])
            dataset_id = int(row["dataset_id"])
            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute
            )
            for i in range(X.shape[1]):
                if type(X.iloc[1, i]) == str:
                    categorical_indicator[i] = True
            # try:
            X = X.drop(X.columns[categorical_indicator], axis=1)
            num_categorical_columns = sum(categorical_indicator)
            original_n_samples, original_n_features = X.shape
            # le = LabelEncoder()
            # y = le.fit_transform(y)

            # X, y, num_pseudo_categorical_columns = remove_pseudo_categorical(X, y)
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
            # categorical_indicator = [categorical_indicator[i] for i in range(len(categorical_indicator)) if
            #                         not i in cols_to_delete]
            print(X.shape)
            print(len(categorical_indicator))

            X, y, num_columns_missing, num_rows_missing, missing_cols_mask = remove_missing_values(X, y)
            print(X.shape)
            print(missing_cols_mask)
            # categorical_indicator = [categorical_indicator[i] for i in range(len(categorical_indicator)) if
            #                         not missing_cols_mask[i]]
            categorical_indicator = [False for i in range(X.shape[1])]
            n_samples, n_features = X.shape
            if pd.isnull(row["Transformation"]):
                transformation = "none"
            else:
                transformation = row["Transformation"]
            y = transform_target(y,transformation)
            train_prop = 0.7
            train_prop = min(15000 / X.shape[0], train_prop)
            numeric_transformer = StandardScaler()
            numeric_transformer_sparse = MaxAbsScaler()
            categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

            # preprocessor = ColumnTransformer(
            #     transformers=[
            #         ("num", numeric_transformer, [i for i in range(X.shape[1]) if not categorical_indicator[i]]),
            #         ("cat", categorical_transformer, [i for i in range(X.shape[1]) if categorical_indicator[i]]),
            #     ]
            # )
            # preprocessor_sparse = ColumnTransformer(
            #     transformers=[
            #         ("num", numeric_transformer_sparse, [i for i in range(X.shape[1]) if not categorical_indicator[i]]),
            #         ("cat", categorical_transformer, [i for i in range(X.shape[1]) if categorical_indicator[i]]),
            #     ]
            # )
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
                        X_train = numeric_transformer.fit_transform(X_train)
                        X_test = numeric_transformer.transform(X_test)
                    except:
                        print("trying MaxAbsScaler")
                        X_train = numeric_transformer_sparse.fit_transform(X_train)
                        X_test = numeric_transformer_sparse.transform(X_test)
                    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
                    y_train, y_test = y_train.astype(np.float32), y_test.astype(
                        np.float32)
                    X_train, X_test = np.array(X_train), np.array(X_test)
                    print(X_train.shape)
                    print(X_test.shape)
                    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
                    if regression:
                        linear_model = TransformedTargetRegressor(regressor=LinearRegression(),
                                                                  transformer=QuantileTransformer(output_distribution="normal"))
                    else:
                        linear_model = LogisticRegression()
                    linear_model.fit(X_train, y_train)
                    if regression:
                        score_linear = -mean_squared_error(y_test, linear_model.predict(X_test), squared=False)  # rsme
                    else:
                        score_linear = linear_model.score(X_test, y_test)  # accuracy
                    #if regression:
                    #    resnet = create_resnet_regressor_skorch(**resnet_config, id=dataset_id)
                    #else:
                    #    resnet = create_resnet_skorch(**resnet_config, id=dataset_id)
                    categorical_indicator = None
                    model, model_id = train_model(iter, X_train, y_train, categorical_indicator, resnet_config)
                    train_score, val_score, score_resnet = evaluate_model(model, X_train, y_train, None, None, X_test,
                                                                        y_test, resnet_config, model_id)
                    score_resnet = -score_resnet #we want high = good so we take -RMSE
                    #if regression:
                    #    score_resnet = -mean_squared_error(y_test, resnet.predict(X_test), squared=False)  # rsme
                    #else:
                    #    score_resnet = resnet.score(X_test, y_test)
                    score_resnet_list.append(score_resnet)
                    score_linear_list.append(score_linear)
                print("Linear score: {}".format(score_linear_list))
                print("Resnet score: {}".format(score_resnet_list))
                if regression:
                    score_linear = np.nanmedian(score_linear_list)
                    score_resnet = np.nanmedian(score_resnet_list)
                else:
                    score_linear = np.mean(score_linear_list)
                    score_resnet = np.mean(score_resnet_list)
                if (score_resnet - score_linear) < - 0.05 * score_resnet:
                    too_easy = True
                else:
                    too_easy = False

                res_dic = {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset.name,
                    "original_n_samples": original_n_samples,
                    "original_n_features": original_n_features,
                    "num_categorical_columns": num_categorical_columns,
                    "num_pseudo_categorical_columns": num_pseudo_categorical_columns,
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

res_df.to_csv("results_resnet_regression.csv")