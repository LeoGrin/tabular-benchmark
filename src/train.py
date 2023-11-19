import numpy as np
from create_models import create_model
import os
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def skorch_evaluation(model, x_train, x_val, x_test, y_train, y_val, y_test, config, model_id, return_r2,
                      delete_checkpoint):
    """
    Evaluate the model
    """
    y_hat_train = model.predict(x_train)
    if x_val is not None:
        y_hat_val = model.predict(x_val)
    y_hat_test = model.predict(x_test)

    # Compute the best train score achieved
    if "regression" in config.keys() and config["regression"]:
        if np.any(np.isnan(y_hat_train)):
            train_score = np.nan
        else:
            if return_r2:
                train_score = r2_score(y_train.reshape(-1), y_hat_train.reshape(-1))
            else:
                train_score = np.sqrt(np.mean((y_hat_train.reshape(-1) - y_train.reshape(-1)) ** 2))
    else:
        train_score = np.sum((y_hat_train == y_train)) / len(y_train)

    if "model__use_checkpoints" in config.keys() and config["model__use_checkpoints"]:
        # check that r"skorch_cp/params_{}.pt".format(model_id) exists
        if os.path.exists(r"skorch_cp/params_{}.pt".format(model_id)):
            print("Using checkpoint")
            if not config["regression"]:
                model.load_params(r"skorch_cp/params_{}.pt".format(model_id))
            else:
                # TransformedTargetRegressor
                if config["transformed_target"]:
                    model.regressor_.load_params(r"skorch_cp/params_{}.pt".format(model_id))
                else:
                    model.load_params(r"skorch_cp/params_{}.pt".format(model_id))
        else:
            print("Checkpoint does not exist")
            print("Outputting NaN")
            return np.nan, np.nan, np.nan

    y_hat_train = model.predict(x_train)
    if x_val is not None:
        y_hat_val = model.predict(x_val)
    y_hat_test = model.predict(x_test)

    if x_val is not None and not np.isnan(y_hat_val).any():
        if "regression" in config.keys() and config["regression"]:
            if return_r2:
                val_score = r2_score(y_val.reshape(-1), y_hat_val.reshape(-1))
            else:
                val_score = np.sqrt(np.mean((y_hat_val.reshape(-1) - y_val.reshape(-1)) ** 2))
        else:
            val_score = np.sum((y_hat_val == y_val)) / len(y_val)
    else:
        val_score = np.nan

    if not np.isnan(y_hat_test).any():
        if "regression" in config.keys() and config["regression"]:
            if return_r2:
                test_score = r2_score(y_test.reshape(-1), y_hat_test.reshape(-1))
            else:
                test_score = np.sqrt(np.mean((y_hat_test.reshape(-1) - y_test.reshape(-1)) ** 2))
        else:
            test_score = np.sum((y_hat_test == y_test)) / len(y_test)
            print(f"Train score checkpoint: {np.sum((y_hat_train == y_train)) / len(y_train)}")
    else:
        test_score = np.nan

    if "model__use_checkpoints" in config.keys() and config["model__use_checkpoints"] and not return_r2 and \
            delete_checkpoint:
        try:
            os.remove(r"skorch_cp/params_{}.pt".format(model_id))
        except:
            print("could not remove params file")
            pass

    return train_score, val_score, test_score


def sklearn_evaluation(fitted_model, x_train, x_val, x_test, y_train, y_val, y_test, config, return_r2):
    """
    Evaluate a fitted model from sklearn
    """
    y_hat_train = fitted_model.predict(x_train)
    y_hat_val = fitted_model.predict(x_val)
    y_hat_test = fitted_model.predict(x_test)

    if "regression" in config.keys() and config["regression"]:
        if return_r2:
            train_score = r2_score(y_train.reshape(-1), y_hat_train.reshape(-1))
        else:
            train_score = np.sqrt(np.mean((y_hat_train.reshape(-1) - y_train.reshape(-1)) ** 2))
    else:
        train_score = np.sum((y_hat_train == y_train)) / len(y_train)

    if "regression" in config.keys() and config["regression"]:
        if return_r2:
            val_score = r2_score(y_val.reshape(-1), y_hat_val.reshape(-1))
        else:
            val_score = np.sqrt(np.mean((y_hat_val.reshape(-1) - y_val.reshape(-1)) ** 2))
    else:
        val_score = np.sum((y_hat_val == y_val)) / len(y_val)

    if "regression" in config.keys() and config["regression"]:
        if return_r2:
            test_score = r2_score(y_test.reshape(-1), y_hat_test.reshape(-1))
        else:
            test_score = np.sqrt(np.mean((y_hat_test.reshape(-1) - y_test.reshape(-1)) ** 2))
    else:
        test_score = np.sum((y_hat_test == y_test)) / len(y_test)

    return train_score, val_score, test_score


def evaluate_model(fitted_model, x_train, y_train, x_val, y_val, x_test, y_test, config, model_id=None, return_r2=False,
                   delete_checkpoint=True):
    """
    Evaluate the model
    """

    if config["model_type"] == "sklearn":
        train_score, val_score, test_score = sklearn_evaluation(fitted_model, x_train, x_val, x_test, y_train, y_val,
                                                                y_test, config, return_r2=return_r2)
    elif config["model_type"] == "skorch":
        train_score, val_score, test_score = skorch_evaluation(fitted_model, x_train, x_val, x_test, y_train, y_val,
                                                               y_test, config, model_id, return_r2=return_r2,
                                                               delete_checkpoint=delete_checkpoint)
    elif config["model_type"] == "tab_survey":
        train_score, val_score, test_score = sklearn_evaluation(fitted_model, x_train, x_val, x_test, y_train, y_val,
                                                                y_test, config, return_r2=return_r2)

    return train_score, val_score, test_score


def train_model(iter, x_train, y_train, categorical_indicator, cat_cardinalities, config, id):
    """
    Train the model
    """
    print("Training")
    if config["model_type"] == "skorch":
        model_raw = create_model(config, categorical_indicator, cat_cardinalities=cat_cardinalities, id=id)  # TODO rng ??
    elif config["model_type"] == "sklearn":
        model_raw = create_model(config, categorical_indicator)
    elif config["model_type"] == "tab_survey":
        model_raw = create_model(config, categorical_indicator, num_features=x_train.shape[1], id=id,
                                 cat_dims=list((x_train[:, categorical_indicator].max(0) + 1).astype(int)))

    if config["regression"] and config["transformed_target"]:
        model = TransformedTargetRegressor(model_raw, transformer=QuantileTransformer(output_distribution="normal"))
    else:
        model = model_raw

    if config["data__categorical"] and "one_hot_encoder" in config.keys() and config[
        "one_hot_encoder"] and not categorical_indicator is None:
        preprocessor = ColumnTransformer([("one_hot", OneHotEncoder(categories="auto", handle_unknown="ignore"),
                                           [i for i in range(x_train.shape[1]) if categorical_indicator[i]]),
                                          ("numerical", "passthrough",
                                           [i for i in range(x_train.shape[1]) if not categorical_indicator[i]])])
        model = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    if config["model_type"] == "tab_survey":
        x_val = x_train[int(len(x_train) * 0.8):]
        y_val = y_train[int(len(y_train) * 0.8):]
        x_train = x_train[:int(len(x_train) * 0.8)]
        y_train = y_train[:int(len(y_train) * 0.8)]
        model.fit(x_train, y_train, x_val, y_val)
    elif config["model_name"].startswith("xgb") and "model__early_stopping_rounds" in config.keys() \
            and config["model__early_stopping_rounds"]:
        x_val = x_train[int(len(x_train) * 0.8):]
        y_val = y_train[int(len(y_train) * 0.8):]
        x_train = x_train[:int(len(x_train) * 0.8)]
        y_train = y_train[:int(len(y_train) * 0.8)]
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
    else:
        if config["model_name"].startswith("tabr"):
            #TODO: handle this elsewhere?
            model.fit({"X": x_train, "y": y_train}, y_train)
        else:
            model.fit(x_train, y_train)

    return model
