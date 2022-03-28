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

os.environ["WANDB_START_METHOD"] = "thread"


def iterate_params_from_possible_params(dic, search_type="grid", n_config=None):
    # takes into input a dic of possible parameters, and iterates on the possible values
    if search_type == "default":  # default except  param speficied, one at a time
        l = []
        for key in dic.keys():
            if key not in ["method_name", "method"]:
                for value in dic[key]:
                    l.append({"method_name": dic["method_name"],
                              "method": dic["method"],
                              key: value})
        return l

    new_dic = {}
    for key in dic.keys():
        if type(dic[key]) != list:
            new_dic[key] = [dic[key]]
        else:
            new_dic[key] = dic[key]
    if search_type == "grid":
        return list(ParameterGrid(new_dic))
    elif search_type == "random":
        return list(ParameterSampler(new_dic, n_iter=n_config))


def merge_all_dics(data_generation_dic, target_generation_dic, data_transforms_list):
    # WARNING: removes "method" key
    # merge all params list into one dictionary with the right prefixes to avoid duplicates
    total_dic = {}
    prefixes = ["data", "target"]
    for i, param_dic in enumerate([data_generation_dic, target_generation_dic]):
        new_param_dic = {prefixes[i] + "__" + "method_name": param_dic["method_name"]}
        for key in param_dic.keys():
            if key not in ["method", "method_name"]:
                new_param_dic[prefixes[i] + "__" + key] = param_dic[key]
        total_dic.update(new_param_dic)
    for i, param_dic in enumerate(data_transforms_list):
        new_param_dic = {"transform__{}__method_name".format(i): param_dic["method_name"]}
        for key in param_dic.keys():
            if key not in ["method", "method_name"]:
                new_param_dic["transform__{}__{}".format(i, key)] = param_dic[key]
        total_dic.update(new_param_dic)

    return total_dic


def create_string_from_dic(dic):
    keys = list(dic.keys())
    keys = np.sort(keys)
    res = ""
    for key in keys:
        res += key + ":" + str(dic[key]) + "/"
    return res

def store_model_function(model, model_name, params_model, params_data, params_target, params_transform_list, config_keyword,
                         x_train, x_test, y_train, y_test):
    prefix = "models/{}/{}/".format(config_keyword, model_name)

    file_name = ".".join("{}:{}".format(k, v) for k, v in list(params_model.items()) if k not in ["method", "method_name", "device"])
    file_name += "--" + ".".join("{}:{}".format(k, v) for k, v in list(params_data.items()) if k not in ["method"])
    file_name += "--" + ".".join("{}:{}".format(k, v) for k, v in list(params_target.items()) if k not in ["method"])
    for params_transform in params_transform_list:
        file_name += "--" + ".".join("{}:{}".format(k, v) for k, v in list(params_transform.items()) if k not in ["method"])

    file_name = prefix + str(hash(prefix + file_name))

    if model_name == "mlp" or model_name == "pretrained":
        model.save_params(f_params="{}.params".format(file_name))
        with open("{}.pkl".format(file_name), "wb") as f:
            pickle.dump(model, f)
    else:
        with open(file_name, "wb") as f:
            pickle.dump(model, f)

    with open(file_name + ".data", "wb") as f:
        pickle.dump((x_train, x_test, y_train, y_test), f)

    return file_name


        # if params_data["method_name"] == "real_data":
        #     model.save_params(f_params="models/{}/{}/{}.pkl".format(config_keyword, params_data["keyword"], model_name))
        #     torch.save(model.module_.state_dict(),
        #                "models/{}/{}/{}.pt".format(config_keyword, params_data["keyword"], model_name))
        #     # saving
        #     with open("models/{}/{}/{}_pickle.pkl".format(config_keyword, params_data["keyword"], model_name),
        #               'wb') as f:
        #         pickle.dump(model, f)
        #     with open("models/{}/{}/{}".format(config_keyword, params_data["keyword"], "data"), 'wb') as f:
        #         pickle.dump((x_train, x_test, y_train, y_test), f)
        #     with open("models/{}/{}/{}_{}".format(config_keyword, params_data["keyword"], model_name, "params"),
        #               'wb') as f:
        #         pickle.dump(params_model, f)
        # else:
        #     if model_name == "rf":
        #         print("saving rf")
        #         with open("models/{}_{}_{}_{}_{}_{}".format(config_keyword, params_data["num_samples"],
        #                                                     params_target["offset"], params_target["period"], iter,
        #                                                     model_name), 'wb') as f:
        #             pickle.dump(model, f)
        #     if model_name == "mlp":
        #         print("saving mlp")
        #         model.save_params(
        #             f_params="models/{}_{}_{}_{}_{}_{}.pkl".format(config_keyword, params_data["num_samples"],
        #                                                            params_target["offset"], params_target["period"],
        #                                                            iter, model_name))
        #         # torch.save(model.module_.state_dict(), "models/{}/{}/{}/{}.pt".format(config_keyword, params_target["offset"], params_target["period"], iter, model_name))
        #         with open("models/{}_{}_{}_{}_{}_{}_pickle.pkl".format(config_keyword, params_data["num_samples"],
        #                                                                params_target["offset"], params_target["period"],
        #                                                                iter, model_name), 'wb') as f:
        #             pickle.dump(model, f)
        #         with open("models/{}_{}_{}_{}_{}_{}".format(config_keyword, params_data["num_samples"],
        #                                                     params_target["offset"], params_target["period"], iter,
        #                                                     "data"), 'wb') as g:
        #             pickle.dump((x_train, x_test, y_train, y_test), g)
        #         with open("models/{}_{}_{}_{}_{}_{}".format(config_keyword, params_data["num_samples"],
        #                                                     params_target["offset"], params_target["period"],
        #                                                     model_name, "params"), 'wb') as f:
        #             pickle.dump(params_model, f)

def evaluate_model(iter, params_model, params_data, params_target, params_transform_list, identifier, use_wandb, run_id,
                   tags, dest, store_model, config_keyword):
    print(iter)
    print(params_model["method_name"])
    n_seconds = 0
    print("SLEEPING FOR {} SECONDS".format(n_seconds))
    time.sleep(n_seconds)
    print(torch.cuda.is_available())
    #print(torch.cuda.current_device())
    #print(torch.cuda.device(torch.cuda.current_device()))
    print(torch.cuda.device_count())
    #print(torch.cuda.get_device_name(torch.cuda.current_device()))
#    print(params_data["keyword"])
    try:
        hash = "".join(map(str, identifier))
        rng = np.random.RandomState(iter)  # can be called without a seed
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
        model_function = params_model["method"]
        model_name = params_model["method_name"]
        params_model_clean = remove_keys_from_dict(params_model, ["method", "method_name"])
        model_name = model_name
        #if model_name[:3] == "nam":
        #



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
                    #    'models/regression_synthetic_{}_{}_{}_{}_mlp_pickle.pkl'.format(5000, 0, 16,
                    #                                                                   iter), 'rb'))
                    #model.initialize()
                    #model.load_params(
                    #    f_params="models/{}_{}_{}_{}_{}_{}.pkl".format(config_keyword, 5000,
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
            model.fit(x_train, y_train)
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
            y_hat_train = model.predict(x_train).reshape(-1)
            y_hat_test = model.predict(x_test).reshape(-1)

            time_elapsed = time.time() - t
            #
            if "regression" in params_data.keys() and params_data["regression"]:
                train_score = np.sqrt(np.mean((y_hat_train - y_train.reshape(-1))**2))
            else:
                train_score = np.sum((y_hat_train == y_train)) / len(y_train)
            #if model_name[:3] == "mlp" or model_name[:3] == "nam" or model_name[:3] == "spa":
            #    model.load_params(r"skorch_cp/params_{}.pt".format(model_id)) #TODO

            #test_score = np.sum((y_hat_test == y_test)) / len(y_test)
            if "regression" in params_data.keys() and params_data["regression"]:
                test_score = np.sqrt(np.mean((y_hat_test - y_test.reshape(-1))**2))
            else:
                test_score = np.sum((y_hat_test == y_test)) / len(y_test)
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
        if model_name[:3] == "mlp" or model_name[:3] == "nam" or model_name[:3] == "spa":
            try:
                os.remove(r"skorch_cp/params_{}.pt".format(model_id))
            except:
                print("could not remove params file")
                pass
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


def grid_search_parallel(path, model_generation_functions, data_generation_functions,
                         target_generation_functions, data_transforms_functions, n_iter=3, search_type="grid",
                         n_config=10, parallel=True, n_jobs=-1, use_wandb=False, run_id=0,
                         tags=["no"], dask=False, dest=None, store_model=False, config_keyword=None):  # TODO use parser
    settings_list = []
    for g, model_generation_dic in enumerate(model_generation_functions):
        for h, params_model in enumerate(
                iterate_params_from_possible_params(model_generation_dic, search_type, n_config)):
            for a, data_generation_dic in enumerate(data_generation_functions):
                for b, params_data in enumerate(iterate_params_from_possible_params(data_generation_dic)):
                    for c, target_generation_dic in enumerate(target_generation_functions):
                        for d, params_target in enumerate(iterate_params_from_possible_params(target_generation_dic)):
                            for e, data_transforms_list in enumerate(data_transforms_functions):
                                params_transform_list_list_raw = [
                                    iterate_params_from_possible_params(data_transforms_dic) for data_transforms_dic in
                                    data_transforms_list]
                                params_transform_list_list = itertools.product(*params_transform_list_list_raw)
                                for f, params_transform_list in enumerate(params_transform_list_list):
                                    identifier = (a, b, c, d, e, f, g, h)
                                    for iter in range(n_iter):
                                        settings_list.append(
                                            [iter, params_model, params_data, params_target, params_transform_list,
                                             identifier, use_wandb, run_id, tags, dest, store_model, config_keyword])
    if parallel:
        print("n_jobs = {}".format(n_jobs))
        if not dask:
            scores = Parallel(n_jobs=n_jobs, verbose=100)(delayed(evaluate_model)(*settings) for settings in settings_list)
        else:
            with joblib.parallel_backend('dask', wait_for_workers_timeout=60):
                scores = Parallel(verbose=100)(delayed(evaluate_model)(*settings) for settings in settings_list)

    else:
        scores = [evaluate_model(*settings) for settings in settings_list]
    # flat_scores = [score for l in scores for score in l]
    df = pd.DataFrame(scores)
    df.to_csv(path)


# def grid_search(models, model_names, n_iter=3, init_id=0):
#     res_dics = []
#     id = init_id
#     for a, data_generation_dic in tqdm.tqdm(enumerate(data_generation_functions)):
#         # print(len(iterate_params_from_possible_params(data_generation_dic)))
#         for params_data in tqdm.tqdm(iterate_params_from_possible_params(data_generation_dic)):
#             for b, target_generation_dic in tqdm.tqdm(enumerate(target_generation_functions)):
#                 # print(len(iterate_params_from_possible_params(target_generation_dic)))
#                 for params_target in tqdm.tqdm(iterate_params_from_possible_params(target_generation_dic)):
#                     for c, data_transforms_dic in enumerate(data_transforms_functions):
#                         # print(len(iterate_params_from_possible_params(data_transforms_dic)))
#                         for params_transform in iterate_params_from_possible_params(data_transforms_dic):
#                             train_scores = np.zeros((len(models), n_iter))
#                             test_scores = np.zeros((len(models), n_iter))
#                             for iter in range(n_iter):
#                                 x, y = generate_dataset([params_data, params_target, [params_transform]])
#                                 x = x.astype(np.float32)  # for skorch
#                                 y = y.astype(np.int64)
#                                 x_train, x_test, y_train, y_test = train_test_split(x, y)
#                                 for i, model in enumerate(models):
#                                     model.fit(x_train, y_train)
#                                     train_scores[i, iter] = model.score(x_train, y_train)
#                                     test_scores[i, iter] = model.score(x_test, y_test)
#                             for i, model in enumerate(models):
#                                 model_name = model_names[i]
#                                 res_dic = {"id": id, "model": model_name,
#                                            "test_scores_mean": np.mean(test_scores, axis=1)[i],
#                                            "test_scores_sd": np.std(test_scores, axis=1)[i],
#                                            "train_scores_mean": np.mean(train_scores, axis=1)[i],
#                                            "train_scores_sd": np.std(train_scores, axis=1)[i]}
#                                 res_dic.update(merge_all_dics(params_data, params_target, params_transform))
#                                 res_dics.append(res_dic)
#                             id += 1
#     df = pd.DataFrame(res_dics)
#     df.to_csv("results.csv")
#     return df


def compare_models_cv(models, dataset_settings, n_iter=1, n_splits=1):
    model_scores = [[] for _ in models]
    for iter in range(n_iter):
        x, y = generate_dataset(dataset_settings)
        print(sum(y))
        x = x.astype(np.float32)  # for skorch
        y = y.astype(np.int64)
        cv = KFold(n_splits=n_splits)
        for i, model in enumerate(models):
            cv_results = cross_validate(model, x, y, cv=cv, return_train_score=True)
            test_scores = cv_results["test_score"]
            train_scores = cv_results["train_score"]
            print(np.mean(train_scores))
            model_scores[i].append(np.mean(test_scores))

    return [np.mean(score) for score in model_scores], [np.std(score) for score in model_scores]


if __name__ == '__main__':
    def create_parser_new():
        parser = argparse.ArgumentParser(description='Argument parser')

        parser.add_argument("-n",
                            "--n-iter",
                            dest="n_iter",
                            help="Number of iteration to run per configuration",
                            type=int,
                            default=30)

        parser.add_argument("--n-configs",
                            dest="n_config",
                            help="Number of configuration run for random search",
                            type=int,
                            default=10)

        parser.add_argument("-config",
                            dest="config_keyword",
                            help="Choose xp to run",
                            type=str,
                            default="sparse")
        parser.add_argument("-dest",
                            dest="dest",
                            help="where to save the results",
                            type=str,
                            default="res")

        parser.add_argument("-search",
                            dest="search_type",
                            help="Grid or random search ?",
                            type=str,
                            default="grid")

        parser.add_argument("-parallel",
                            dest="parallel",
                            help="Use joblib",
                            type=bool,
                            default=True)

        parser.add_argument('-wandb', dest='use_wandb', action='store_true')
        parser.set_defaults(use_wandb=False)

        parser.add_argument("--n-jobs",
                            dest="n_jobs",
                            help="Number of jobs for joblib",
                            type=int,
                            default=-1)

        parser.add_argument('-dask', dest='dask', action='store_true')
        parser.set_defaults(dask=False)

        parser.add_argument('--store-model', dest='store_model', action='store_true')
        parser.set_defaults(store_model=False)

        parser.add_argument("--tags", nargs="+", default=["no tag"])

        return parser


    parser = create_parser_new()
    args = parser.parse_args()

    model_generation_functions, data_generation_functions, target_generation_functions, transform_generation_functions = config(
        args.config_keyword)

    if args.dask:
        from dask_jobqueue import SLURMCluster



        cluster = SLURMCluster(cores=1,
                              #processes=1,
                               memory='4G',
                               walltime='10:00:00',
                               queue = "electronic,funky,jazzy",
                               #queue="normal,parietal")
                              extra = ['--resources GPU=1'])
        cluster.scale(jobs=args.n_jobs)
        #cluster.adapt(maximum_jobs=args.n_jobs)
        #cluster.adapt(maximum_jobs=args.n_jobs)
        from dask.distributed import Client
        client = Client(cluster)
        #print("PRINTING")
        #print(client.get_versions(check=True))
        #print(client.scheduler_info()['workers'])
        print("TEST")
        print(client.submit(lambda x: x + 1, 10).result())
        print("LINK")
        print(client.dashboard_link)


    else:
        client = None

    if args.store_model:
        try:
            os.mkdir("models/{}".format(args.config_keyword))
        except FileExistsError:
            pass


    grid_search_parallel("{}.csv".format(args.dest), model_generation_functions, data_generation_functions, #TODO results
                         target_generation_functions, transform_generation_functions,
                         n_iter=args.n_iter, search_type=args.search_type, n_config=args.n_config,
                         parallel=args.parallel, n_jobs=args.n_jobs, use_wandb=args.use_wandb,
                         run_id=int(time.time()), tags=args.tags, dask=args.dask, dest=args.dest, store_model=args.store_model,
                         config_keyword=args.config_keyword)
