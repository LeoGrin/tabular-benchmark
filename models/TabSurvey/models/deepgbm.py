from models.basemodel_torch import BaseModelTorch

from models.deepgbm_lib.main import train, predict
from models.deepgbm_lib.preprocess.preprocessing_cat import CatEncoder
from models.deepgbm_lib.preprocess.preprocessing_num import NumEncoder
import models.deepgbm_lib.config as deepgbm_config

'''
    DeepGBM: A Deep Learning Framework Distilled by GBDT for Online Prediction Tasks
    (https://www.microsoft.com/en-us/research/publication/deepgbm-a-deep-learning-framework-distilled-by-gbdt-for-online-prediction-tasks/)
    
    Code adapted from: https://github.com/motefly/DeepGBM
'''


class DeepGBM(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "classification":
            print("DeepGBM not implemented for classification!")
            import sys
            sys.exit()

        if args.cat_idx:
            cat_col = args.cat_idx
            num_col = list(set(range(args.num_features)) - set(args.cat_idx))
        else:
            cat_col = []
            num_col = list(range(args.num_features))

        self.ce = CatEncoder(cat_col, num_col)
        self.ne = NumEncoder(cat_col, num_col)

        deepgbm_config.update({'task': args.objective,
                               "epochs": args.epochs,
                               "early-stopping": args.early_stopping_rounds,
                               "batch_size": args.batch_size,
                               "test_batch_size": args.val_batch_size,
                               "device": self.device})
        deepgbm_config.update(**params)

        print(deepgbm_config)

    def fit(self, X, y, X_val=None, y_val=None):
        # preprocess
        train_x_cat, feature_sizes = self.ce.fit_transform(X.copy())
        test_x_cat = self.ce.transform(X_val.copy())

        train_x = self.ne.fit_transform(X)
        test_x = self.ne.transform(X_val)

        train_num = (train_x, y.reshape(-1, self.args.num_classes))
        test_num = (test_x, y_val.reshape(-1, self.args.num_classes))

        # train
        self.model, _, loss_history, val_loss_history = train(train_num, test_num, train_x_cat.astype('int32'),
                                                              test_x_cat.astype('int32'), feature_sizes)

        return loss_history, val_loss_history

    def predict_helper(self, X):
        return predict(self.model, X, self.ce, self.ne).reshape(-1, 1)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "n_trees": trial.suggest_categorical("n_trees", [100, 200]),
            "maxleaf": trial.suggest_categorical("maxleaf", [64, 128]),
            "loss_de": trial.suggest_int("loss_de", 2, 10),
            "loss_dr": trial.suggest_categorical("loss_dr", [0.7, 0.9])
        }
        return params
