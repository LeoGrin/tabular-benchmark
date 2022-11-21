from models.basemodel import BaseModel
from utils.io_utils import get_output_path

import numpy as np
import os

import tensorflow as tf
from sklearn.metrics import log_loss

from models.dnf_lib.DNFNet.ModelHandler import ModelHandler, EarlyStopping, ReduceLRonPlateau
from models.dnf_lib.config import get_config
from models.dnf_lib.Utils.NumpyGenerator import NumpyGenerator
from models.dnf_lib.Utils.experiment_utils import create_model, create_experiment_directory


'''
    Net-DNF: Effective Deep Modeling of Tabular Data  (https://openreview.net/forum?id=73WTGs96kho)
    
    Code adapted from: https://github.com/amramabutbul/DisjunctiveNormalFormNet
'''


class DNFNet(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            print("DNF-Net not implemented for regression tasks.")
            import sys
            sys.exit(0)

        self.config, self.score_config = get_config(args.dataset, 'DNFNet')

        self.config.update({
            'input_dim': args.num_features,
            'output_dim': args.num_classes,
            'translate_label_to_one_hot': True if args.objective == "classification" else False,
            'epochs': args.epochs,
            'early_stopping_patience': args.early_stopping_rounds,
            'batch_size': args.batch_size,
            'GPU': str(args.gpu_ids),
            **self.params
        })

        self.score_config.update({
            'score_metric': log_loss,
            'score_increases': False,
        })

        print(self.config)

        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['GPU']
        tf.reset_default_graph()
        tf.random.set_random_seed(seed=self.config['random_seed'])
        np.random.seed(seed=self.config['random_seed'])
        self.score_metric = self.score_config['score_metric']

        self.experiment_dir, self.weights_dir, self.logs_dir = create_experiment_directory(self.config,
                                                                                           return_sub_dirs=True)
        self.model = create_model(self.config, models_module_name=self.config['models_module_name'])

        self.model_handler = None

    def fit(self, X, y, X_val=None, y_val=None):
        train_generator = NumpyGenerator(X, y, self.config['output_dim'],
                                         self.config['batch_size'],
                                         translate_label_to_one_hot=self.config['translate_label_to_one_hot'],
                                         copy_dataset=False)
        val_generator = NumpyGenerator(X_val, y_val, self.config['output_dim'],
                                       self.config['batch_size'],
                                       translate_label_to_one_hot=self.config['translate_label_to_one_hot'],
                                       copy_dataset=False)

        early_stopping = EarlyStopping(patience=self.config['early_stopping_patience'],
                                       score_increases=self.score_config['score_increases'], monitor='val_score')
        lr_scheduler = ReduceLRonPlateau(initilal_lr=self.config['initial_lr'], factor=self.config['lr_decay_factor'],
                                         patience=self.config['lr_patience'], min_lr=self.config['min_lr'],
                                         monitor='train_loss')

        self.model_handler = ModelHandler(config=self.config, model=self.model,
                                          callbacks=[lr_scheduler, early_stopping],
                                          target_dir=self.weights_dir, logs_dir=self.logs_dir)
        self.model_handler.build_graph(phase='train')
        loss, val_loss = self.model_handler.train(train_generator, val_generator, score_metric=self.score_metric,
                                                  score_increases=self.score_config['score_increases'])

        return loss, val_loss

    def predict_proba(self, X):
        assert os.path.exists(self.weights_dir + '/model_weights.ckpt.meta')

        if os.path.exists(self.weights_dir + '/model_weights.ckpt.meta'):
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(self.model_handler.sess, self.weights_dir + '/model_weights.ckpt')

        # create sorted target array for Generator to work and to sort the predictions afterwards
        y = np.array(list(range(X.shape[0])))
        if self.args.objective == "classification":
            r = np.zeros((X.shape[0], self.args.num_classes - 1))
            y = np.concatenate([y.reshape(-1, 1), r], axis=1)

        test_generator = NumpyGenerator(X, y, self.config['output_dim'], self.config['batch_size'],
                                        translate_label_to_one_hot=False,  # self.config['translate_label_to_one_hot']
                                        copy_dataset=False)

        y, y_pred = self.model_handler.test(test_generator)

        # Sort the predictions!
        y_pred_sorted = [y_pred for _, y_pred in sorted(zip(y[:, 0], y_pred))]

        self.prediction_probabilities = np.concatenate(y_pred_sorted, axis=0).reshape(-1, self.args.num_classes)

        if self.args.objective == "binary":
            self.prediction_probabilities = np.concatenate((1 - self.prediction_probabilities,
                                                            self.prediction_probabilities), 1)

        return self.prediction_probabilities

    def save_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="")

        # Model already saved, only coping to the right location
        import shutil

        if os.path.exists(filename):
            shutil.rmtree(filename)

        shutil.copytree(self.weights_dir, filename)  # , dirs_exist_ok=True

    def get_model_size(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print(total_parameters)

        return total_parameters


    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "n_formulas": trial.suggest_categorical("n_formulas", [64, 128, 256, 512, 1024]),  # 2048, 3072
            "elastic_net_beta": trial.suggest_categorical("elastic_net_beta", [1.6, 1.3, 1., 0.7, 0.4, 0.1]),
        }
        return params
