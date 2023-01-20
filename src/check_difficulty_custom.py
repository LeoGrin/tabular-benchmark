import sklearn.datasets
from check_difficulty import check_difficulty
from preprocessing.preprocessing import preprocessing
import traceback
import pandas as pd
import numpy as np
import argparse

def import_data(filename, target_variable, categorical_columns=[],
                numerical_columns=[], datetime_columns=[], to_drop_colums=[],
                sep=None, name=None):
    print("###########################################")
    print("Importing {}".format(name))
    if sep is None:
        df = pd.read_csv(filename)
    else:
        df = pd.read_csv(filename, sep=sep)
    original_n_samples, original_n_features = df.shape
    original_n_features -= 1  # remove the target column
    assert len(categorical_columns) == 0 or len(numerical_columns) == 0
    if len(numerical_columns) > 0:
        categorical_columns = [col for col in df.columns if col not in numerical_columns and col != target_variable]

    x_ = df.drop(to_drop_colums, axis=1)
    x_ = x_.drop([target_variable], axis=1)
    for col in datetime_columns:
         x_[col] = pd.to_numeric(pd.to_datetime(x_[col]), errors='coerce')
    y_ = df[target_variable]

    categorical_indicator = [col in categorical_columns for col in x_.columns]
    return x_, y_, categorical_indicator, original_n_samples, original_n_features

def check_difficulty_custom(name, X, y, categorical_indicator, categorical=False, regression=False, transformation="none", device="cpu",
                            remove_model=[]):
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

    if regression:
        resnet_config["model_name"] = "rtdl_resnet_regressor"
        resnet_config["regression"] = True
        resnet_config["data__regression"] = True
        resnet_config["transformed_target"] = True
    else:
        resnet_config["model_name"] = "rtdl_resnet"
        resnet_config["regression"] = False
        resnet_config["data__regression"] = False
        resnet_config["transformed_target"] = False
    if categorical:
        resnet_config["data__categorical"] = True
    else:
        resnet_config["data__categorical"] = False

    try:
        X, y, categorical_indicator, num_high_cardinality, num_columns_missing, num_rows_missing, \
        num_categorical_columns, n_pseudo_categorical, original_n_samples, original_n_features = \
            preprocessing(X, y, categorical_indicator, categorical=categorical,
                          regression=regression, transformation=transformation)
        res_dic = {
            "dataset_id": pd.NA,
            "dataset_name": name,
            "original_n_samples": original_n_samples,
            "original_n_features": original_n_features,
            "num_categorical_columns": num_categorical_columns,
            "num_pseudo_categorical_columns": n_pseudo_categorical,
            "num_high_cardinality": num_high_cardinality,
            "num_columns_missing": num_columns_missing,
            "num_rows_missing": num_rows_missing,
            "score_resnet": pd.NA,
            "score_linear": pd.NA,
            "score_hgbt": pd.NA,
            "score_tree": pd.NA,
            "heterogeneous": pd.NA,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "too_small": pd.NA,
            "too_many_features": pd.NA,
            "not_enough_categorical": pd.NA}


        if X.shape[1] > 2000 or X.shape[0] <= 3000 or X.shape[1] <= 3:
            if X.shape[1] > 2000:
                res_dic["too_many_features"] = True
            else:
                res_dic["too_small"] = True
        elif categorical and num_categorical_columns < 1:
            res_dic["not_enough_categorical"] = True
        else:
            score_dic = check_difficulty(X, y, categorical_indicator, categorical, regression,
                                         resnet_config=resnet_config,
                                         remove_model=remove_model)
            for key in score_dic.keys():
                res_dic[key] = score_dic[key]
    except:
        print("ERROR")
        print(traceback.format_exc())
        res_dic = {
            "dataset_id": pd.NA,
            "dataset_name": name,
            "original_n_samples": pd.NA,
            "original_n_features": pd.NA,
            "num_categorical_columns": pd.NA,
            "num_pseudo_categorical_columns": pd.NA,
            "num_high_cardinality": pd.NA,
            "num_columns_missing": pd.NA,
            "num_rows_missing": pd.NA,
            "score_resnet": pd.NA,
            "score_linear": pd.NA,
            "score_hgbt": pd.NA,
            "score_tree": pd.NA,
            "heterogeneous": pd.NA,
            "n_samples": pd.NA,
            "n_features": pd.NA,
            "too_small": pd.NA,
            "too_many_features": pd.NA,
            "not_enough_categorical": pd.NA}

    return res_dic

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Train a model on a dataset')

    # Add the arguments
    parser.add_argument('--device', type=str, default="cuda", help='Device to use')
    parser.add_argument('--file', type=str, default="filename", help='Csv with all datasets')
    parser.add_argument('--datasets', nargs='+', type=int, default=[], help='Datasets to use')
    parser.add_argument('--out_file', type=str, default="filename", help='filename to save')
    parser.add_argument('--regression', action='store_true', help='True if regression, false otherwise')
    parser.add_argument('--categorical', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--all', action='store_true', help="Whether to check all datasets or only those already "
                                                           "deemed too easy with a HGBT")
    parser.add_argument('--remove_model', nargs='+', help='List of models not to try')

    # Parse the arguments
    args = parser.parse_args()

    if args.remove_model is None:
        args.remove_model = []


    #########
    # Numerical regression
    #########

    res_df = pd.DataFrame()

    bunch = sklearn.datasets.fetch_california_housing(as_frame=True)
    x, y = bunch.data, bunch.target
    res_dic = check_difficulty_custom("california", x, y, categorical_indicator=np.zeros(x.shape[1], dtype=bool),
                            categorical=False, regression=True, transformation="log", device=args.device,
                                      remove_model=args.remove_model)
    
    
    res_df = res_df.append(res_dic, ignore_index=True)
    
    
    data = pd.read_csv("../data/raw/players_22.csv")
    #print(data.shape)
    #data.columns = ['fixed acidity', "volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
    original_n_samples, original_n_features = data.shape
    original_n_features -= 1 #remove the target column
    y = data["wage_eur"]
    num_cols = ['age', 'height_cm', 'weight_kg', 'league_level',
           'international_reputation', 'release_clause_eur', 'club_contract_valid_until',
           'club_joined']
    x = data[num_cols]
    x.loc[:, 'club_joined'] = pd.to_numeric(pd.to_datetime(x['club_joined']), errors='coerce')
    
    res_dic = check_difficulty_custom("fifa", x, y, categorical_indicator=np.zeros(x.shape[1], dtype=bool),
                            categorical=False, regression=True, transformation="log",
                                      device=args.device, remove_model=args.remove_model)
    res_df = res_df.append(res_dic, ignore_index=True)

    data = pd.read_csv("../data/raw/YearPredictionMSD.txt", header=None)
    original_n_samples, original_n_features = data.shape
    original_n_features -= 1 #remove the target column
    print(data)
    y = data[0]
    x = data.drop(0, axis=1)
    
    res_dic = check_difficulty_custom("year", x, y, categorical_indicator=np.zeros(x.shape[1], dtype=bool),
                            categorical=False, regression=True, transformation="none",
                                      device=args.device, remove_model=args.remove_model)
    res_df = res_df.append(res_dic, ignore_index=True)

    data = pd.read_csv("../data/raw/yahoo_valid_1.csv")
    original_n_samples, original_n_features = data.shape
    original_n_features -= 1 #remove the target column
    y = data["0"]
    print(np.unique(y, return_counts=True))
    print(x.columns)
    x = data.drop(["Unnamed: 0", "0"], axis=1)
    
    
    res_dic = check_difficulty_custom("yahoo", x, y, categorical_indicator=np.zeros(x.shape[1], dtype=bool),
                            categorical=False, regression=True, transformation="none",
                                      device=args.device, remove_model=args.remove_model)
    res_df = res_df.append(res_dic, ignore_index=True)

    # Save the results
    res_df.to_csv("difficulty_custom_numerical_regression_3.csv", index=False)

    #########
    # Numerical classification
    #########

    # res_df = pd.DataFrame()

    #
    # df = pd.read_csv("../data/raw/heloc_dataset_v1.csv")
    # original_n_samples, original_n_features = df.shape
    # original_n_features -= 1 #remove the target column
    # y = df["RiskPerformance"]
    # x = df.drop("RiskPerformance", axis=1)
    #
    # res_dic = check_difficulty_custom("heloc", x, y, categorical_indicator=np.zeros(x.shape[1], dtype=bool),
    #                         categorical=False, regression=False, transformation="none",
    #                                   device=args.device, remove_model=args.remove_model)
    # res_df = res_df.append(res_dic, ignore_index=True)
    #
    #
    #
    # df = pd.read_csv("../data/raw/Churn_Modelling.csv")
    # original_n_samples, original_n_features = df.shape
    # original_n_features -= 1 #remove the target column
    # x = df[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]]
    # y = df["Exited"]
    # res_dic = check_difficulty_custom("churn", x, y, categorical_indicator=np.zeros(x.shape[1], dtype=bool),
    #                                   categorical=False, regression=False, transformation="none",
    #                                   device=args.device, remove_model=args.remove_model)
    # res_df = res_df.append(res_dic, ignore_index=True)
    #
    #
    #
    # df = pd.read_csv("../data/raw/online_shoppers_intention.csv")
    # original_n_samples, original_n_features = df.shape
    # original_n_features -= 1 #remove the target column
    # x = df[['Administrative', 'Administrative_Duration', 'Informational',
    #        'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
    #        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']]
    # y = df["Revenue"]
    #
    # res_dic = check_difficulty_custom("shopping", x, y, categorical_indicator=np.zeros(x.shape[1], dtype=bool),
    #                                   categorical=False, regression=False, transformation="none",
    #                                   device=args.device, remove_model=args.remove_model)
    # res_df = res_df.append(res_dic, ignore_index=True)
    #
    #
    #
    #
    # bunch = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/ozone/eighthr.data")
    # bunch = bunch.replace("?", np.NaN)
    # bunch = bunch.to_numpy()
    # y = bunch[:,-1]
    # x = bunch[:,:-1]
    # x = pd.DataFrame(x)
    # res_dic = check_difficulty_custom("ozone", x, y, categorical_indicator=np.zeros(x.shape[1], dtype=bool),
    #                                   categorical=False, regression=False, transformation="none",
    #                                   device=args.device, remove_model=args.remove_model)
    # res_df = res_df.append(res_dic, ignore_index=True)
    #
    #
    #
    # df = pd.read_csv("../data/raw/GiveMeSomeCredit/cs-training.csv")
    # original_n_samples, original_n_features = df.shape
    # original_n_features -= 1 #remove the target column
    # x = df.drop(["SeriousDlqin2yrs", 'Unnamed: 0'], axis=1) #Q: do I want to remove integer variables?
    # y = df["SeriousDlqin2yrs"]
    # res_dic = check_difficulty_custom("credit", x, y, categorical_indicator=np.zeros(x.shape[1], dtype=bool),
    #                                   categorical=False, regression=False, transformation="none",
    #                                   device=args.device, remove_model=args.remove_model)
    # res_df = res_df.append(res_dic, ignore_index=True)
    #
    #
    #
    #
    # bunch = sklearn.datasets.fetch_california_housing(as_frame=True)
    # x, y = bunch.data, bunch.target
    # original_n_samples, original_n_features = x.shape
    # y = y > np.median(y)
    # res_dic = check_difficulty_custom("california", x, y, categorical_indicator=np.zeros(x.shape[1], dtype=bool),
    #                                   categorical=False, regression=False, transformation="none",
    #                                   device=args.device, remove_model=args.remove_model)
    # res_df = res_df.append(res_dic, ignore_index=True)
    #
    #
    # bunch = sklearn.datasets.fetch_covtype(as_frame=True)
    # x, y = bunch.data, bunch.target
    # original_n_samples, original_n_features = x.shape
    # num_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
    # x = x[num_cols]
    # mask = (y == 1) | (y == 2)
    # y = y[mask]
    # x = x[mask]
    # y = (y == 1)
    # try:
    #     res_dic = check_difficulty_custom("covtype", x, y, categorical_indicator=np.zeros(x.shape[1], dtype=bool),
    #                                   categorical=False, regression=False, transformation="none",
    #                                   device=args.device, remove_model=args.remove_model)
    # except:
    #     print("covtype failed")
    #     print("ERROR")
    #     pass
    # res_df = res_df.append(res_dic, ignore_index=True)
    #
    #
    # data = pd.read_csv("../data/raw/players_22.csv")
    # #print(data.shape)
    # #data.columns = ['fixed acidity', "volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
    # original_n_samples, original_n_features = data.shape
    # original_n_features -= 1 #remove the target column
    # y = data["wage_eur"]
    # num_cols = ['age', 'height_cm', 'weight_kg', 'league_level',
    #        'international_reputation', 'release_clause_eur', 'club_contract_valid_until',
    #        'club_joined']
    # x = data[num_cols]
    # x['club_joined'] = pd.to_numeric(pd.to_datetime(x['club_joined']), errors='coerce')
    #
    # y = y > np.nanmedian(y)
    # res_dic = check_difficulty_custom("fifa", x, y, categorical_indicator=np.zeros(x.shape[1], dtype=bool),
    #                                   categorical=False, regression=False, transformation="none",
    #                                   device=args.device, remove_model=args.remove_model)
    # res_df = res_df.append(res_dic, ignore_index=True)
    #
    # # Save results
    # res_df.to_csv("results_custom_numerical_classif.csv", index=False)

    ############
    # Categorical classification
    ############

    # res_df = pd.DataFrame()
    # x, y, categorical_indicator, original_n_samples, original_n_features = import_data("../data/raw/Churn_Modelling.csv",
    #                       target_variable="Exited",
    #                      numerical_columns=["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"],
    #                      name = "churn")
    # res_dic = check_difficulty_custom("churn", x, y, categorical_indicator=categorical_indicator,
    #                                     categorical=True, regression=False, transformation="none",
    #                                     device=args.device, remove_model=args.remove_model)
    # res_dic["original_n_samples"] = original_n_samples
    # res_dic["original_n_features"] = original_n_features
    #
    # res_df = res_df.append(res_dic, ignore_index=True)
    #
    #
    #
    # x, y, categorical_indicator, original_n_samples, original_n_features = import_data("../data/raw/online_shoppers_intention.csv",
    #                       target_variable="Revenue",
    #                       numerical_columns=['Administrative', 'Administrative_Duration', 'Informational',
    #         'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
    #         'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'],
    #                       name = "shopping")
    #
    # res_dic = check_difficulty_custom("shopping", x, y, categorical_indicator=categorical_indicator,
    #                               categorical=True, regression=False, transformation="none",
    #                               device=args.device, remove_model=args.remove_model)
    # res_dic["original_n_samples"] = original_n_samples
    # res_dic["original_n_features"] = original_n_features
    # res_df = res_df.append(res_dic, ignore_index=True)


    # x, y, categorical_indicator, original_n_samples, original_n_features = import_data("../data/raw/cox-violent-parsed.csv",
    #                       target_variable="is_recid",
    #                       numerical_columns = ['age', 'juv_fel_count', "juv_misd_count","juv_other_count", "priors_count",
    #                                             "days_b_screening_arrest", "c_days_from_compas", "r_days_from_arrest",
    #                                             "c_days_from_compas", "end"],
    #                       to_drop_colums= ["violent_recid", "is_violent_recid", "event",
    #               'decile_score', "type_of_assessment"],
    #                       datetime_columns=['compas_screening_date', 'dob', "c_jail_in",
    #                       "c_jail_out", "c_offense_date", "c_arrest_date", "r_offense_date", "r_jail_in", "r_jail_out"],
    #                       name = "compass")
    # res_dic = check_difficulty_custom("compass", x, y, categorical_indicator=categorical_indicator,
    #                                   categorical=True, regression=False, transformation="none",
    #                                   device=args.device, remove_model=args.remove_model)
    # res_dic["original_n_samples"] = original_n_samples
    # res_dic["original_n_features"] = original_n_features
    # res_df = res_df.append(res_dic, ignore_index=True)

    #
    # x, y, categorical_indicator, original_n_samples, original_n_features = import_data("../data/raw/support2.csv",
    #                       target_variable="death",
    #                       numerical_columns = ["age", "slos", "d.time", "edu", "scoma", "charges", "totcst",
    #                      "avtisst", "hday", "meanbp", "wblc", "hrt",
    #                      "resp", "temp", "pafi", "alb", "bili", "crea",
    #                      "sod", "ph", "glucose", "bun", "urine", "adlp", "adls", "adlsc"],
    #                       to_drop_colums= ["sps", "aps", "surv2m", "surv6m", "prg2m", "prg6m", "dnrday"],
    #                       name = "support",
    #                       sep=";")
    # res_dic = check_difficulty_custom("support", x, y, categorical_indicator=categorical_indicator,
    #                                   categorical=True, regression=False, transformation="none",
    #                                   device=args.device, remove_model=args.remove_model)
    # res_dic["original_n_samples"] = original_n_samples
    # res_dic["original_n_features"] = original_n_features
    # res_df = res_df.append(res_dic, ignore_index=True)
    #
    # x, y, categorical_indicator, original_n_samples, original_n_features = import_data("../data/raw/Telco-Customer-Churn.csv",
    #                       target_variable="Churn",
    #                       numerical_columns = ["tenure", "MonthlyCharges",
    #                      "TotalCharges"],
    #                       name = "blastchar")
    # res_dic = check_difficulty_custom("blastchar", x, y, categorical_indicator=categorical_indicator,
    #                                   categorical=True, regression=False, transformation="none",
    #                                   device=args.device, remove_model=args.remove_model)
    # res_dic["original_n_samples"] = original_n_samples
    # res_dic["original_n_features"] = original_n_features
    # res_df = res_df.append(res_dic, ignore_index=True)

    # Save results
    # res_df.to_csv("results_custom_categorical_classif_2.csv", index=False)




    # res_dic = import_data("../data/raw/store.csv",
    #                       target_variable="Churn",
    #                       numerical_columns = ["tenure", "MonthlyCharges",
    #                      "TotalCharges"],
    #                       name = "blastchar")
    # res_df = res_df.append(res_dic, ignore_index=True)
    #
    #
    # df = pd.read_csv("../data/raw/store.csv")
    # original_n_samples, original_n_features = df.shape
    # original_n_features -= 1  # remove the target column
    # x = df.drop(["Churn"], axis=1)
    # y = df["Churn"]
    # print(y)
    # numerical_columns = []
    # numerical_columns = ["tenure", "MonthlyCharges",
    #                      "TotalCharges"]
    #
    # categorical_indicator = [col not in numerical_columns for col in x.columns]
    #
    # x, y, num_columns_missing, num_rows_missing, num_pseudo_categorical_columns, score_hbgt, score_logistic, too_easy, checks_passed = transform_and_save(
    #     x, y, categorical_indicator, "telco_churn")
    # n_samples, n_features = x.shape
    # save_names(x, "shopping")



    #
    # data = pd.read_csv("../data/raw/players_22.csv")
    # #print(data.shape)
    # #data.columns = ['fixed acidity', "volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
    # original_n_samples, original_n_features = data.shape
    # original_n_features -= 1 #remove the target column
    # x = data.drop(["wage_eur"], axis=1)
    # y = data["wage_eur"]
    # numerical_columns = ['age', 'height_cm', 'weight_kg', 'league_level',
    #        'international_reputation', 'release_clause_eur', 'club_contract_valid_until',
    #        'club_joined']
    # x['club_joined'] = pd.to_numeric(pd.to_datetime(x['club_joined']), errors='coerce')
    # categorical_indicator = [col not in numerical_columns for col in x.columns]
    #
    # y = y > np.nanmedian(y)
    # x, y, num_columns_missing, num_rows_missing, num_pseudo_categorical_columns, score_hbgt, score_logistic, too_easy, checks_passed = transform_and_save(x, y, "fifa")
    # #save_names(x, "fifa")
    # n_samples, n_features = x.shape
    #
    #
    # res_dic = {
    #     "dataset_id": pd.NA,
    #     "dataset_name": "fifa",
    #     "original_n_samples": original_n_samples,
    #     "original_n_features": original_n_features,
    #     "n_samples": n_samples,
    #     "n_features": n_features,
    #     "num_categorical_columns": pd.NA,
    #     "num_pseudo_categorical_columns": num_pseudo_categorical_columns,
    #     "num_columns_missing": num_columns_missing,
    #     "num_rows_missing": num_rows_missing,
    #     "too_easy": too_easy,
    #     "score_hbgt": score_hbgt,
    #     "score_logistic": score_logistic,
    #     "checks_passed": checks_passed,
    #     "heterogeneous": pd.NA}
    # res_df = res_df.append(res_dic, ignore_index=True)