import itertools
import os
from generate_dataset_pipeline import generate_dataset, apply_transform
from utils.utils import numpy_to_dataset, remove_keys_from_dict
from utils.plot_utils import plot_decision_boudaries
from sklearn.tree import export_graphviz
from generate_data import *
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, KFold, ParameterGrid, ParameterSampler
from config import config
from joblib import Parallel, delayed
import argparse
import time
import sys, traceback  # Needed for pulling out your full stackframe info
import wandb
import joblib
from data_transforms import select_features_rf
import pickle
import torch
from utils.keyword_to_function_conversion import convert_keyword_to_function
from train import *

def evaluate_model(config):
    try:
        #rng = np.random.RandomState(iter)  #FIXME
        x_train, x_test, y_train, y_test = generate_dataset(config)#, rng) #FIXME

        model = create_model(config)

        model = train_model(model, x_train, y_train, x_test, y_test, config)

        train_score, test_score = evaluate_model(model, x_train, y_train, x_test, y_test, config)

        if config["use_wandb"]:
                wandb_run.finish()
            if model_name == "rf" and "store_features" in tags:
                print(params_data["keyword"])
                features_used_array = np.zeros((len(model.estimators_), x_train.shape[1]))
                for i, estimator in enumerate(model.estimators_):
                    features_used = [feature for feature in estimator.tree_.feature if feature != -2]
                    features_used, features_count = np.unique(features_used, return_counts=True)
                    #features_used = features_names[features_used]
                    features_used_array[i] = features_count
                with open("features/{}".format(params_data["keyword"]), "wb") as f:
                    pickle.dump(features_used_array, f)


            # try:
            #     with open("data/numerical_only/names_{}".format(params_data["keyword"]), "rb") as f:
            #         features_names = pickle.load(f)
            #     for i in range(params_model_dic["n_estimators"]):
            #         tree = model.estimators_[i]
            #         export_graphviz(tree,
            #                     out_file='trees/{}_{}_tree.dot'.format(params_data["keyword"], i),
            #                     feature_names=features_names,
            #                     class_names=list(map(str, range(2))),
            #                     rounded=True, proportion=False,
            #                     precision=2, filled=True)
            # except:
            #     pass
            if not no_fit:
                y_hat_train = model.predict(x_train).reshape(-1)
                y_hat_test = model.predict(x_test).reshape(-1)

                time_elapsed = time.time() - t
                #
                if "regression" in params_data.keys() and params_data["regression"]:
                    train_score = np.sqrt(np.mean((y_hat_train - y_train.reshape(-1))**2))
                else:
                    train_score = np.sum((y_hat_train == y_train)) / len(y_train)
                if "use_checkpoints" in params_model.keys() and params_model["use_checkpoints"]:
                    model.load_params(r"skorch_cp/params_{}.pt".format(model_id)) #TODO

                #test_score = np.sum((y_hat_test == y_test)) / len(y_test)
                if "regression" in params_data.keys() and params_data["regression"]:
                    test_score = np.sqrt(np.mean((y_hat_test - y_test.reshape(-1))**2))
                else:
                    test_score = np.sum((y_hat_test == y_test)) / len(y_test)
            else:
                train_score = np.nan
                test_score = np.nan
                time_elapsed = np.nan
            print("#########")
            print(time.time() - t)
            print(params_model)
            print(params_data)
            print(train_score)
            print(test_score)
            print("###########")
            res_dic.update({"test_scores": test_score,
                            "train_scores": train_score,
                            "time_elapsed": time_elapsed,
                            "n_train": len(y_train),
                            "n_test": len(y_test),
                            "n_features": x_train.shape[1]})
            if store_model:
                res_dic["model_file_name"] = stored_model_file_name
            #pd.DataFrame(res_dic).to_csv("results")
            if model_name == "rf":
                print("rf")
                print(model.feature_importances_)
        except:
            res_dic.update({"test_scores": np.nan,
                            "train_scores": np.nan,
                            "time_elapsed": np.nan})
            print("ERROR DURING FIT")
            print(params_model)
            print(traceback.format_exc())

        res_dic.update(remove_keys_from_dict(params_model, ["method"]))
        res_dic.update({"data_generation_str": create_string_from_dic(all_params)})
        res_dic.update({"model_params_str": create_string_from_dic(remove_keys_from_dict(params_model, ["method"]))})
        res_dic.update(all_params)
        if "use_checkpoints" in params_model.keys() and params_model["use_checkpoints"]:
            try:
                os.remove(r"skorch_cp/params_{}.pt".format(model_id))
            except:
                print("could not remove params file")
                pass
        #del model
        # dic_plot_train = {"y_pred": y_hat_train, "y_true": y_train}
        # dic_plot_test = {"y_pred": y_hat_test, "y_true": y_test}
        # for i in range(x.shape[1]):
        #    dic_plot_train[names[i]] = x_train[:, i]
        #    dic_plot_test[names[i].format(i)] = x_test[:, i]
        # pd.DataFrame(dic_plot_train).to_csv("predictions/{}_predictions_{}_train.csv".format(model_name, params_data["keyword"]))
        # pd.DataFrame(dic_plot_test).to_csv("predictions/{}_predictions_{}_test.csv".format(model_name, params_data["keyword"]))

    except:
        # Print to the console
        print("ERROR OUTSIDE FIT")
        # To get the traceback information
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(traceback.format_exc())
        print(params_data, params_target, params_transform_list)
        return {}

    return res_dic