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

def evaluate_model(config):
    try:
        #rng = np.random.RandomState(iter)  #FIXME
        x, y = generate_dataset([params_data, params_target, params_transform_list], rng)
        #print(y.dtype)
        #x, y, features = select_features_rf(x, y, rng, 2, return_features=True)
        #with open("data/numerical_only/names_{}".format(params_data["keyword"]), 'rb') as f:
        #  names = pickle.load(f)
        #names = np.array(names)[features]
       # print(names)

        #x = x[:, [0, 2]]
        x = x.astype(np.float32)  # for skorch
        all_params = merge_all_dics(params_data, params_target, params_transform_list)
        res_dic = {"iter": iter, "id": hash}
        #model_function = params_model["method"]
        model_name = params_model["method_name"]
        model_function = convert_keyword_to_function(model_name)
        params_model_clean = remove_keys_from_dict(params_model, ["method_name"])
        #if model_name[:3] == "nam":
        #


        #TODO: split should be done in the data generation
        n_rows = x.shape[0]
        if "max_num_samples" in params_data.keys():
            train_set_size = min(params_data["max_num_samples"] / n_rows, 0.75)
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_set_size, random_state=rng)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=rng)


        x_train, x_test, y_train, y_test = apply_transform(x_train, x_test, y_train, y_test, params_transform_list, rng)
        x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32) #for skorch

        if "regression" in params_data.keys() and params_data["regression"]:
            y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
            y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)
        else:
            y_train, y_test = y_train.astype(np.compat.long), y_test.astype(np.compat.long)



        try:
            if model_name[:3] == "mlp" or model_name[:3] == "nam" or model_name[:3] == "spa" or model_name == "pretrained":
                if use_wandb:
                    config = all_params
                    config["run_id"] = run_id
                    config["model_name"] = model_name
                    config["iter"] = iter
                    config["dim"] = x_train.shape[1]
                    config["n_train"] = x_train.shape[0]
                    config.update(params_model_clean)
                    wandb_run = wandb.init(project="ToyTabular",
                                           entity="leogrin",
                                           # notes="tweak baseline",
                                           # tags=["baseline", "paper1"),
                                           config=config,
                                           tags=tags)

                else:
                    wandb_run = None
                # y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)
                if model_name[:3] == "mlp" or model_name == "pretrained":
                    model_id = dest + "_" +  hash + str(params_model["hidden_size"]) + str(iter)
                    print("kwargs")
                    print(params_model_clean)
                    model = model_function(model_id, wandb_run,
                                           **params_model_clean)  # create a different checkpointing file for each run to avoid conflict (I'm not 100% sure it's necessary)
                    #model = pickle.load(open(
                    #    'saved_models/regression_synthetic_{}_{}_{}_{}_mlp_pickle.pkl'.format(5000, 0, 16,
                    #                                                                   iter), 'rb'))
                    #model.initialize()
                    #model.load_params(
                    #    f_params="saved_models/{}_{}_{}_{}_{}_{}.pkl".format(config_keyword, 5000,
                     #                                                  params_target["offset"], params_target["period"],
                    #                                                   iter, model_name))
#                    model.module_.no_reinitialize = True
                    #state_dict = model.module_.state_dict()
                    # Add noise
                    #for param in state_dict.keys():
                    #    print("#########")
                    #    if param.startswith("fc_layers") or param.startswith("input_layer") or param.startswith("output_layer"):
                    #        state_dict[param] = model.module_.state_dict()[param] + 0 * torch.randn(model.module_.state_dict()[param].shape)
                    #        model.module_.load_state_dict(state_dict)
                elif model_name[:3] == "nam" or model_name[:3] == "spa":
                    model_id = dest + "_" + hash + str(iter)  # TODO
                    model = model_function(model_id, wandb_run,
                                           **params_model_clean)  # create a different checkpointing file for each run to avoid conflict (I'm not 100% sure it's necessary)
            else:
                model = model_function(**params_model_clean)

            t = time.time()
            print("fitting....")
            print(x_train.shape)
            print(y_train.shape)
            #y_hat_train_0 = model.predict(x_train).reshape(-1)

            #if model_name[:3] != "mlp":
            #print("fitting")
            #print(model.module_.no_reinitialize)
            #model.partial_fit(x_train, y_train)
            if not no_fit:
                model.fit(x_train, y_train)
            else:
                time.sleep(10)
            #else:
            #    print("f")
                #model.warm_start = True
                #model.fit(x_train, y_train, epochs=0)
            #    print(model.initialized_)
            #    model.partial_fit(x_train, y_train, epochs=0)
            #model.fit(x_train, y_train)

            if store_model:
                stored_model_file_name = store_model_function(model, model_name, params_model, params_data, params_target, params_transform_list,
                            config_keyword, x_train, x_test, y_train, y_test)
                print("model name")
                print(stored_model_file_name)

            #plot_decision_boudaries(x_train, y_train, x_test, y_test, model, title=model_name)
            if use_wandb:
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