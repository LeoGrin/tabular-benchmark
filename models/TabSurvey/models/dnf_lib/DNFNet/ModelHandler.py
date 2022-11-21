import os
import shutil

import numpy as np
import tensorflow as tf
from models.dnf_lib.Utils.NumpyGenerator import NumpyGenerator
from models.dnf_lib.Utils.experiment_utils import create_model, create_experiment_directory


def get_if_exists(dict, key):
    if key in dict:
        return dict[key]
    return None


def score_comparator_wrapper(score_increases):
    if score_increases:
        return lambda a, b: a > b
    else:
        return lambda a, b: a < b


class EarlyStopping(object):
    def __init__(self, patience, score_increases, monitor='val_score'):
        if monitor == 'val_loss':
            score_increases = False

        self.patience = patience
        self.best_test_score = np.NINF if score_increases else np.inf
        self.score_comparator = score_comparator_wrapper(score_increases)
        self.score_not_improved_counter = 0
        self.monitor = monitor

    def _early_stop(self, score):
        if self.score_comparator(score, self.best_test_score):
            self.best_test_score = score
            self.score_not_improved_counter = 0
            return False

        self.score_not_improved_counter += 1
        if self.score_not_improved_counter >= self.patience:
            print("Early Stopping")
            return True
        return False

    def pre_training(self, model):
        pass

    def epoch_end(self, model):
        if self.monitor == 'val_score':
            model.early_stopping = self._early_stop(model.val_score)
        elif self.monitor == 'val_loss':
            model.early_stopping = self._early_stop(model.val_loss)


class ReduceLRonPlateau(object):
    def __init__(self, initilal_lr, factor, patience, min_lr, monitor='val_loss'):
        self.lr = initilal_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_loss = np.inf
        self.loss_not_reducing_counter = 0
        self.monitor = monitor

    def compute_lr(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.loss_not_reducing_counter = 0
            return self.lr

        self.loss_not_reducing_counter += 1
        if self.loss_not_reducing_counter >= self.patience:
            self.loss_not_reducing_counter = 0
            self.lr = self.lr * self.factor
            if self.lr < self.min_lr:
                self.lr = self.min_lr

            print("Learning rate: {}".format(self.lr))
        return self.lr

    def pre_training(self, model):
        model.current_lr = self.lr

    def epoch_end(self, model):
        if self.monitor == 'val_loss':
            model.current_lr = self.compute_lr(model.val_loss)
        elif self.monitor == 'train_loss':
            model.current_lr = self.compute_lr(model.train_loss)
        else:
            print('ReduceLRonPlateau: {} is not supported'.format(self.monitor))


class ModelHandler:
    def __init__(self, config, model, callbacks, target_dir, logs_dir=None):
        self.model = model
        self.config = config
        self.target_dir = target_dir
        self.logs_dir = logs_dir
        self.callbacks = callbacks
        self.n_labels = self.config['output_dim']
        self.sess = None
        self.current_lr = self.config['initial_lr']
        self.early_stopping = None
        self.val_loss = None
        self.val_score = None
        self.train_loss = None

        # placeholders
        self.x = None
        self.y = None
        self.y_weights = None
        self.lr = None
        self.dropout_rate = None
        self.is_training = None

        # returned vars from the model
        self.output = None
        self.features = None
        self.loss = None
        self.optimizer = None
        self.labels = None

    def define_placeholders(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.config['input_dim']], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None, self.config['output_dim']], name="y")
        self.y_weights = tf.placeholder(tf.float32, shape=[None, 1], name="y_weights")
        self.lr = tf.placeholder(tf.float32, name="lr")
        self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        placeholders = {
            'x': self.x,
            'y': self.y,
            'y_weights': self.y_weights,
            'lr': self.lr,
            'dropout_rate': self.dropout_rate,
            'is_training': self.is_training,
        }
        return placeholders

    def build_graph(self, load_weights=False, phase='train'):
        placeholders = self.define_placeholders()
        build_dict = self.model.build(placeholders, n_labels=self.n_labels, phase=phase)
        self.optimizer = get_if_exists(build_dict, 'optimizer')
        self.loss = get_if_exists(build_dict, 'loss')
        self.output = get_if_exists(build_dict, 'output')
        self.features = get_if_exists(build_dict, 'features')
        self.labels = get_if_exists(build_dict, 'labels')

        self.sess = tf.Session()

        if load_weights:
            print('Loading weights')
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(self.sess, self.target_dir + '/model_weights.ckpt')

    def _translate_feed_dict(self, str_feed_dict):
        return {getattr(self, filed_name): value for filed_name, value in str_feed_dict.items()}

    def callbacks_handler(self, op):
        for callback in self.callbacks:
            if op == 'pre_training':
                callback.pre_training(self)
            elif op == 'epoch_end':
                callback.epoch_end(self)

    def train(self, train_generator, val_generator, score_metric, score_increases):
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        score_comparator = score_comparator_wrapper(score_increases)
        best_val_score = np.NINF if score_increases else np.inf
        self.early_stopping = False
        self.callbacks_handler(op='pre_training')

        loss_history = []
        val_loss_history = []

        try:
            for epoch in range(self.config['epochs']):

                train_loss_sum = 0
                while train_generator.get_epochs() <= epoch:
                    str_feed_dict = train_generator.get_batch()
                    feed_dict = self._translate_feed_dict(str_feed_dict)
                    feed_dict[self.lr] = self.current_lr
                    feed_dict[self.is_training] = True
                    _, l = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                    train_loss_sum += l * str_feed_dict['x'].shape[0]
                self.train_loss = train_loss_sum / train_generator.get_dataset_size()
                loss_history.append(self.train_loss)

                val_loss_sum = 0
                y_true = None
                y_pred = None
                while val_generator.get_epochs() <= epoch:
                    str_feed_dict = val_generator.get_batch()
                    feed_dict = self._translate_feed_dict(str_feed_dict)
                    feed_dict[self.lr] = self.current_lr
                    feed_dict[self.is_training] = False
                    l, pred = self.sess.run([self.loss, self.output], feed_dict=feed_dict)
                    val_loss_sum += l * str_feed_dict['x'].shape[0]
                    y_batch = str_feed_dict['y']
                    y_true = y_batch if y_true is None else np.concatenate([y_true, y_batch])
                    y_pred = pred if y_pred is None else np.concatenate([y_pred, pred])
                self.val_loss = val_loss_sum / val_generator.get_dataset_size()
                val_loss_history.append(self.val_loss)

                self.val_score = score_metric(y_true, y_pred)
                assert val_generator.get_dataset_size() == y_true.shape[0]

                print("Epoch: {0:}, loss: {1:.6f}, val loss: {2:.6f}, score: {3:.6f}".format(epoch, self.train_loss,
                                                                                             self.val_loss,
                                                                                             self.val_score))
                if score_comparator(self.val_score, best_val_score):
                    print("Val score improved from {} to {}".format(best_val_score, self.val_score))
                    best_val_score = self.val_score
                    if self.config['save_weights'] and epoch >= self.config['starting_epoch_to_save']:
                        print('saving model weights.')
                        saver.save(self.sess, os.path.join(self.target_dir, "model_weights.ckpt"))

                self.callbacks_handler(op='epoch_end')

                if self.early_stopping:
                    break

            print("Best validation score: {}".format(best_val_score))
            # return best_val_score, epoch
            return loss_history, val_loss_history

        except tf.errors.ResourceExhaustedError:
            return None, epoch

    def test(self, test_generator):
        test_generator.reset()
        y = None
        y_pred = None
        while test_generator.get_epochs() < 1:
            str_feed_dict = test_generator.get_batch()
            feed_dict = self._translate_feed_dict(str_feed_dict)
            feed_dict[self.is_training] = False
            pred = self.sess.run(self.output, feed_dict=feed_dict)
            y_batch = str_feed_dict['y']
            y = y_batch if y is None else np.concatenate([y, y_batch])
            y_pred = pred if y_pred is None else np.concatenate([y_pred, pred])

        assert test_generator.get_dataset_size() == y.shape[0]

        return y, y_pred

    @staticmethod
    def train_and_test(config, data, score_config):
        print('train {}'.format(config['model_name']))
        os.environ["CUDA_VISIBLE_DEVICES"] = config['GPU']
        tf.reset_default_graph()
        tf.random.set_random_seed(seed=config['random_seed'])
        np.random.seed(seed=config['random_seed'])
        score_metric = score_config['score_metric']

        experiment_dir, weights_dir, logs_dir = create_experiment_directory(config, return_sub_dirs=True)
        model = create_model(config, models_module_name=config['models_module_name'])
        train_generator = NumpyGenerator(data['X_train'], data['Y_train'], config['output_dim'], config['batch_size'],
                                         translate_label_to_one_hot=config['translate_label_to_one_hot'],
                                         copy_dataset=False)
        val_generator = NumpyGenerator(data['X_val'], data['Y_val'], config['output_dim'], config['batch_size'],
                                       translate_label_to_one_hot=config['translate_label_to_one_hot'],
                                       copy_dataset=False)
        test_generator = NumpyGenerator(data['X_test'], data['Y_test'], config['output_dim'], config['batch_size'],
                                        translate_label_to_one_hot=config['translate_label_to_one_hot'],
                                        copy_dataset=False)

        early_stopping = EarlyStopping(patience=config['early_stopping_patience'],
                                       score_increases=score_config['score_increases'], monitor='val_score')
        lr_scheduler = ReduceLRonPlateau(initilal_lr=config['initial_lr'], factor=config['lr_decay_factor'],
                                         patience=config['lr_patience'], min_lr=config['min_lr'], monitor='train_loss')

        model_handler = ModelHandler(config=config, model=model, callbacks=[lr_scheduler, early_stopping],
                                     target_dir=weights_dir, logs_dir=logs_dir)
        model_handler.build_graph(phase='train')
        val_score, epoch = model_handler.train(train_generator, val_generator, score_metric=score_metric,
                                               score_increases=score_config['score_increases'])

        assert os.path.exists(model_handler.target_dir + '/model_weights.ckpt.meta')

        if os.path.exists(model_handler.target_dir + '/model_weights.ckpt.meta'):
            print('Loading weights')
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(model_handler.sess, model_handler.target_dir + '/model_weights.ckpt')

        y_true, y_pred = model_handler.test(test_generator)
        test_score = score_metric(y_true, y_pred)
        print('Test score: {}'.format(test_score))

        model_handler.sess.close()
        shutil.rmtree(weights_dir, ignore_errors=True)
        return {'test_score': test_score, 'validation_score': val_score, 'n_epochs': epoch}
