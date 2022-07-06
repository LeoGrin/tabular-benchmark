import os
os.environ["PROJECT_DIR"] = "test"
from generate_dataset_pipeline import generate_dataset
from train import *

if __name__ == "__main__":

    # config = {"model_name": "rf_c",
    #           "model_type": "sklearn",
    #           "data__method_name": "real_data",
    #           "data__keyword": "california",
    #           "transform__0__method_name": "gaussienize",
    #           "transform__0__type": "quantile",
    #           #"transform__1__method_name": "select_features_rf",
    #           #"transform__1__num_features": 2,
    #           "transform__1__method_name": "remove_high_frequency_from_train",
    #           "transform__1__cov_mult": 2,
    #           "train_prop": 0.70,
    #           "max_train_samples": 10000,
    #           "val_test_prop": 0.3,
    #           "max_val_samples": 50000,
    #          "max_test_samples": 50000}

    config = {"model_name": "ft_transformer_regressor",
                      "model_type": "skorch",
                      "model__use_checkpoints": False,
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
                      #"model__verbose": 100,
                      "model__max_epochs": 2,
                      "regression": True,
                      "data__regression": True,
                      "data__method_name": "real_data",
                      "data__categorical": True,
                      "data__keyword": "yprop_4_1",
                      "transform__0__method_name": "gaussienize",
                      "transform__0__type": "quantile",
                      "transform__0__apply_on": "numerical",
                      "transformed_target": True,
                      "max_train_samples": 10000}

    config = {"model_name": "ft_transformer_regressor",
                      "model_type": "skorch",
                      "model__use_checkpoints": False,
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
                      #"model__verbose": 100,
                      "model__max_epochs": 2,
                      "regression": True,
                      "data__regression": True,
                      "data__method_name": "real_data",
                      "data__categorical": False,
                      "data__keyword": "superconduct",
                      "transform__0__method_name": "gaussienize",
                      "transform__0__type": "quantile",
                      "transform__0__apply_on": "numerical",
                      "transformed_target": True,
                      "max_train_samples": 10000}

    CONFIG_DEFAULT = {"train_prop": 0.70,
                      "val_test_prop": 0.3,
                      "max_val_samples": 50000,
                      "max_test_samples": 50000}

    config.update(CONFIG_DEFAULT)

    rng = np.random.RandomState(32)
    x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator = generate_dataset(config, rng)
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
    args = AttrDict()
    params = AttrDict()
    config = {"regression": True,
                      "data__regression": True,
                      "data__method_name": "real_data",
                      "data__categorical": False,
                      "data__keyword": "superconduct",
                      "transform__0__method_name": "gaussienize",
                      "transform__0__type": "quantile",
                      "transform__0__apply_on": "numerical",
                      "transformed_target": True,
                      "max_train_samples": 10000,
                        "model_type": "tab_survey",
                "model__args__batch_size": 128,
                 "model__args__val_batch_size": 128,
                 "model__args__epochs": 2,
                 "model__args__early_stopping_rounds": 10,
                "model__args__num_features": x_train.shape[1],
                "model__args__use_gpu": False,
                 "model__args__data_parallel": False,
                 "model__args__cat_idx": [],
                 "model__args__num_classes": 1,
                 "model__args__objective": "regression",
                 "model__args__model_name": 'saint',
                 "model_name": "saint",
                 "model__args__dataset": "superconduct",
                 "model__params__depth": 6,
                 "model__params__heads": 8,
                 "model__params__dropout": 0.1}
    model, model_id = train_model(0, x_train, y_train, categorical_indicator, config)
    #saint = SAINT(args=args,
    #              params=params)
    #saint.fit(x_train, y_train, x_val, y_val)
    #y_hat = saint.predict(x_test)

    # if config["model_type"] == "skorch" and config["regression"] == True:
    #     print("YES")
    #     y_train, y_val, y_test = y_train.reshape(-1, 1), y_val.reshape(-1, 1), y_test.reshape(-1, 1)
    #     y_train, y_val, y_test = y_train.astype(np.float32), y_val.astype(np.float32), y_test.astype(
    #         np.float32)
    # else:
    #     y_train, y_val, y_test = y_train.reshape(-1), y_val.reshape(-1), y_test.reshape(-1)
    #     # y_train, y_val, y_test = y_train.astype(np.float32), y_val.astype(np.float32), y_test.astype(np.float32)
    # x_train, x_val, x_test = x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(
    #     np.float32)
    #
    #
    # model, model_id = train_model(0, x_train, y_train, categorical_indicator, config)
    # r2_train, r2_val, r2_test = evaluate_model(model, x_train, y_train, x_val, y_val, x_test,
    #                                                                    y_test, config, model_id, return_r2=True)
    # train_score, val_score, test_score = evaluate_model(model, x_train, y_train, x_val, y_val, x_test,
    #                                                     y_test, config, model_id)
    #
    #
    # print(r2_train, r2_val, r2_test)
    # print(train_score, val_score, test_score)
    #
    #

