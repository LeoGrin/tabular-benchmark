import numpy as np
from create_models import create_model
import os



def skorch_evaluation(model, x_train, x_val, x_test, y_train, y_val, y_test, config, model_id):
    """
    Evaluate the model
    """
    y_hat_train = model.predict(x_train)
    y_hat_val = model.predict(x_val)
    y_hat_test = model.predict(x_test)

    if "regression" in config.keys() and config["regression"]:
        train_score = np.sqrt(np.mean((y_hat_train - y_train.reshape(-1)) ** 2))
    else:
        train_score = np.sum((y_hat_train == y_train)) / len(y_train)

    if "use_checkpoints" in config.keys() and config["use_checkpoints"]:
        model.load_params(r"skorch_cp/params_{}.pt".format(model_id))  # TODO

    if "regression" in config.keys() and config["regression"]:
        val_score = np.sqrt(np.mean((y_hat_val - y_val.reshape(-1)) ** 2))
    else:
        val_score = np.sum((y_hat_val == y_val)) / len(y_val)

    if "regression" in config.keys() and config["regression"]:
        test_score = np.sqrt(np.mean((y_hat_test - y_test.reshape(-1)) ** 2))
    else:
        test_score = np.sum((y_hat_test == y_test)) / len(y_test)

    if "use_checkpoints" in config.keys() and config["use_checkpoints"]:
        try:
            os.remove(r"skorch_cp/params_{}.pt".format(model_id))
        except:
            print("could not remove params file")
            pass



    return train_score, val_score, test_score

def sklearn_evaluation(fitted_model, x_train, x_val, x_test, y_train, y_val, y_test, config):
    """
    Evaluate a fitted model from sklearn
    """
    y_hat_train = fitted_model.predict(x_train)
    y_hat_val = fitted_model.predict(x_val)
    y_hat_test = fitted_model.predict(x_test)

    if "regression" in config.keys() and config["regression"]:
        train_score = np.sqrt(np.mean((y_hat_train - y_train.reshape(-1)) ** 2))
    else:
        train_score = np.sum((y_hat_train == y_train)) / len(y_train)

    if "regression" in config.keys() and config["regression"]:
        val_score = np.sqrt(np.mean((y_hat_val - y_val.reshape(-1)) ** 2))
    else:
        val_score = np.sum((y_hat_val == y_val)) / len(y_val)

    if "regression" in config.keys() and config["regression"]:
        test_score = np.sqrt(np.mean((y_hat_test - y_test.reshape(-1)) ** 2))
    else:
        test_score = np.sum((y_hat_test == y_test)) / len(y_test)

    return train_score, val_score, test_score

def evaluate_model(fitted_model, x_train, y_train, x_val, y_val, x_test, y_test, config, model_id=None):
    """
    Evaluate the model
    """

    if config["model_type"] == "sklearn":
        train_score, val_score, test_score = sklearn_evaluation(fitted_model, x_train, x_val, x_test, y_train, y_val, y_test, config)
    elif config["model_type"] == "skorch":
        train_score, val_score, test_score = skorch_evaluation(fitted_model, x_train, x_val, x_test, y_train, y_val, y_test, config, model_id)

    return train_score, val_score, test_score

def train_model(iter, x_train, y_train, config):
    """
    Train the model
    """
    if config["model_type"] == "skorch":
        id = ".".join(list(config.keys())) + "." + str(iter) # uniquely identify the run (useful for checkpointing)
        model = create_model(config, id)  # TODO rng ??
        print(x_train.shape)
        print(y_train.shape)
        model.fit(x_train, y_train)
    elif config["model_type"] == "sklearn":
        id = None
        model = create_model(config)
        model.fit(x_train, y_train)

    return model, id