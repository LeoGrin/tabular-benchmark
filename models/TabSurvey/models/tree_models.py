import xgboost as xgb
import catboost as cat
import lightgbm as lgb

import numpy as np

from models.basemodel import BaseModel

'''
    Define all Gradient Boosting Decision Tree Models:
    XGBoost, CatBoost, LightGBM
'''

'''
    XGBoost (https://xgboost.readthedocs.io/en/stable/)
'''


class XGBoost(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        self.params["verbosity"] = 1

        if args.use_gpu:
            self.params["tree_method"] = "gpu_hist"
            self.params["gpu_id"] = args.gpu_ids[0]

        if args.objective == "regression":
            self.params["objective"] = "reg:squarederror"
            self.params["eval_metric"] = "rmse"
        elif args.objective == "classification":
            self.params["objective"] = "multi:softprob"
            self.params["num_class"] = args.num_classes
            self.params["eval_metric"] = "mlogloss"
        elif args.objective == "binary":
            self.params["objective"] = "binary:logistic"
            self.params["eval_metric"] = "auc"

    def fit(self, X, y, X_val=None, y_val=None):
        train = xgb.DMatrix(X, label=y)
        val = xgb.DMatrix(X_val, label=y_val)
        eval_list = [(val, "eval")]
        self.model = xgb.train(self.params, train, num_boost_round=self.args.epochs, evals=eval_list,
                               early_stopping_rounds=self.args.early_stopping_rounds,
                               verbose_eval=self.args.logging_period)

        return [], []

    def predict(self, X):
        X = xgb.DMatrix(X)
        return super().predict(X)

    def predict_proba(self, X):
        probabilities = self.model.predict(X)

        if self.args.objective == "binary":
            probabilities = probabilities.reshape(-1, 1)
            probabilities = np.concatenate((1 - probabilities, probabilities), 1)

        self.prediction_probabilities = probabilities
        return self.prediction_probabilities

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 12, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True)
        }
        return params


'''
    CatBoost (https://catboost.ai/)
'''


class CatBoost(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        self.params["iterations"] = self.args.epochs
        self.params["od_type"] = "Iter"
        self.params["od_wait"] = self.args.early_stopping_rounds
        self.params["verbose"] = self.args.logging_period
        self.params["train_dir"] = "output/CatBoost/" + self.args.dataset + "/catboost_info"

        if args.use_gpu:
            self.params["task_type"] = "GPU"
            self.params["devices"] = [self.args.gpu_ids]

        self.params["cat_features"] = self.args.cat_idx

        if args.objective == "regression":
            self.model = cat.CatBoostRegressor(**self.params)
        elif args.objective == "classification" or args.objective == "binary":
            self.model = cat.CatBoostClassifier(**self.params)

    def fit(self, X, y, X_val=None, y_val=None):

        # CatBoost does not accept float arrays if cat features are defined
        if self.args.cat_idx:
            X = X.astype('object')
            X_val = X_val.astype('object')
            X[:, self.args.cat_idx] = X[:, self.args.cat_idx].astype('int')
            X_val[:, self.args.cat_idx] = X_val[:, self.args.cat_idx].astype('int')

        self.model.fit(X, y, eval_set=(X_val, y_val))

        return [], []

    def predict(self, X):
        if self.args.cat_idx:
            X = X.astype('object')
            X[:, self.args.cat_idx] = X[:, self.args.cat_idx].astype('int')

        return super().predict(X)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 12, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
        }
        return params


'''
    LightGBM (https://lightgbm.readthedocs.io/en/latest/)
'''


class LightGBM(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        self.params["verbosity"] = -1

        if args.objective == "regression":
            self.params["objective"] = "regression"
            self.params["metric"] = "mse"
        elif args.objective == "classification":
            self.params["objective"] = "multiclass"
            self.params["num_class"] = args.num_classes
            self.params["metric"] = "multiclass"
        elif args.objective == "binary":
            self.params["objective"] = "binary"
            self.params["metric"] = "auc"

    def fit(self, X, y, X_val=None, y_val=None):
        train = lgb.Dataset(X, label=y, categorical_feature=self.args.cat_idx)
        val = lgb.Dataset(X_val, label=y_val, categorical_feature=self.args.cat_idx)
        self.model = lgb.train(self.params, train, num_boost_round=self.args.epochs, valid_sets=[val],
                               valid_names=["eval"], callbacks=[lgb.early_stopping(self.args.early_stopping_rounds),
                                                                lgb.log_evaluation(self.args.logging_period)],
                               categorical_feature=self.args.cat_idx)

        return [], []

    def predict_proba(self, X):
        probabilities = self.model.predict(X)

        if self.args.objective == "binary":
            probabilities = probabilities.reshape(-1, 1)
            probabilities = np.concatenate((1 - probabilities, probabilities), 1)

        self.prediction_probabilities = probabilities
        return self.prediction_probabilities

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 2, 4096, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        }
        return params
