from sklearn.preprocessing import OneHotEncoder

from models.basemodel import BaseModel

from keras.callbacks import Callback, EarlyStopping
from keras import backend as K
from pandas import DataFrame
import numpy as np

from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense

from utils.io_utils import get_output_path

'''
    Regularization Learning Networks: Deep Learning for Tabular Datasets (https://arxiv.org/abs/1805.06440)

    Code adapted from: https://github.com/irashavitt/regularization_learning_networks
'''
class RLN(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        lr = np.power(10, self.params["log_lr"])
        build_fn = self.RLN_Model(layers=self.params["layers"], norm=self.params["norm"],
                                  avg_reg=self.params["theta"], learning_rate=lr)

        arguments = {
            'build_fn': build_fn,
            'epochs': args.epochs,
            'batch_size': self.args.batch_size,
            'verbose': 1,
        }

        if args.objective == "regression":
            self.model = KerasRegressor(**arguments)
        else:
            self.model = KerasClassifier(**arguments)

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X).astype('float32')
        X_val = np.asarray(X_val).astype('float32')

        if self.args.objective == "classification":
            # Needs the classification targets one hot encoded
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
            y = ohe.fit_transform(y.reshape(-1, 1))
            y_val = ohe.transform(y_val.reshape(-1, 1))

        history = self.model.fit(X, y, validation_data=(X_val, y_val))
        # Early Stopping has to be defined in the RLN_Model method

        return history.history["loss"], history.history["val_loss"]

    def predict(self, X):
        X = np.asarray(X).astype('float32')
        return super().predict(X)

    def predict_proba(self, X):
        X = np.asarray(X).astype('float32')
        return super().predict_proba(X)

    def save_model(self, filename_extension="", directory="models"):
        filename = get_output_path(self.args, directory=directory, filename="m", extension=filename_extension,
                                   file_type="h5")
        self.model.model.save(filename)

    def get_model_size(self):
        return 0

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "layers": trial.suggest_int("layers", 2, 8),
            "theta": trial.suggest_int("theta", -12, -8),
            "log_lr": trial.suggest_int("log_lr", 5, 7),
            "norm": trial.suggest_categorical("norm", [1, 2]),
        }
        return params

    """
        Adapted methods from the original implementation from 
        https://github.com/irashavitt/regularization_learning_networks
    """

    def RLN_Model(self, layers=4, **rln_kwargs):
        def build_fn():
            model = self.base_model(layers=layers)()

            # print(model.summary())

            # For efficiency we only regularized the first layer
            rln_callback = RLNCallback(model.layers[0], **rln_kwargs)

            # Change the fit function of the model to except rln_callback:
            orig_fit = model.fit

            def rln_fit(*args, **fit_kwargs):
                # orig_callbacks = fit_kwargs.get('callbacks', [])
                # Has to be set in here or Keras will crash...
                orig_callbacks = [EarlyStopping(patience=self.args.early_stopping_rounds)]

                rln_callbacks = orig_callbacks + [rln_callback]
                return orig_fit(*args, callbacks=rln_callbacks, **fit_kwargs)

            model.fit = rln_fit

            return model

        return build_fn

    def base_model(self, layers=4, l1=0.01):
        assert layers > 1

        INPUT_DIM = self.args.num_features
        OUTPUT_DIM = self.args.num_classes

        if self.args.objective == "regression":
            loss_fn = "mse"
            act_fn = None
        elif self.args.objective == "classification":
            loss_fn = "categorical_crossentropy"
            act_fn = "softmax"
        elif self.args.objective == "binary":
            loss_fn = "binary_crossentropy"
            act_fn = "sigmoid"

        def build_fn():
            inner_l1 = l1
            # create model
            model = Sequential()
            # Construct the layers of the model to form a geometric series
            prev_width = INPUT_DIM
            for width in np.exp(np.log(INPUT_DIM) * np.arange(layers - 1, 0, -1) / layers):
                width = int(np.round(width))
                model.add(Dense(width, input_dim=prev_width, kernel_initializer='glorot_normal', activation='relu',
                                kernel_regularizer=regularizers.l1(inner_l1)))
                # For efficiency we only regularized the first layer
                inner_l1 = 0
                prev_width = width

            model.add(Dense(OUTPUT_DIM, kernel_initializer='glorot_normal', activation=act_fn))

            # Compile model
            model.compile(loss=loss_fn, optimizer='Adam')
            return model

        return build_fn


class RLNCallback(Callback):
    def __init__(self, layer, norm=1, avg_reg=-7.5, learning_rate=6e5):
        """
        An implementation of Regularization Learning, described in https://arxiv.org/abs/1805.06440, as a Keras
        callback.
        :param layer: The Keras layer to which we apply regularization learning.
        :param norm: Norm of the regularization. Currently supports only l1 and l2 norms. Best results were obtained
        with l1 norm so far.
        :param avg_reg: The average regularization coefficient, Theta in the paper.
        :param learning_rate: The learning rate of the regularization coefficients, nu in the paper. Note that since we
        typically have many weights in the network, and we optimize the coefficients in the log scale, optimal learning
        rates tend to be large, with best results between 10^4-10^6.
        """
        super(RLNCallback, self).__init__()
        self._kernel = layer.kernel
        self._prev_weights, self._weights, self._prev_regularization = [None] * 3
        self._avg_reg = avg_reg
        self._shape = K.transpose(self._kernel).get_shape().as_list()
        self._lambdas = DataFrame(np.ones(self._shape) * self._avg_reg)
        self._lr = learning_rate
        assert norm in [1, 2], "Only supporting l1 and l2 norms at the moment"
        self.norm = norm

    def on_train_begin(self, logs=None):
        self._update_values()

    def on_batch_end(self, batch, logs=None):
        self._prev_weights = self._weights
        self._update_values()
        gradients = self._weights - self._prev_weights

        # Calculate the derivatives of the norms of the weights
        if self.norm == 1:
            norms_derivative = np.sign(self._weights)
        else:
            norms_derivative = self._weights * 2

        if self._prev_regularization is not None:
            # This is not the first batch, and we need to update the lambdas
            lambda_gradients = gradients.multiply(self._prev_regularization)
            self._lambdas -= self._lr * lambda_gradients

            # Project the lambdas onto the simplex \sum(lambdas) = Theta
            translation = (self._avg_reg - self._lambdas.mean().mean())
            self._lambdas += translation

        # Clip extremely large lambda values to prevent overflow
        max_lambda_values = np.log(np.abs(self._weights / norms_derivative)).fillna(np.inf)
        self._lambdas = self._lambdas.clip(upper=max_lambda_values)

        # Update the weights
        regularization = norms_derivative.multiply(np.exp(self._lambdas))
        self._weights -= regularization
        K.set_value(self._kernel, self._weights.values.T)
        self._prev_regularization = regularization

    def _update_values(self):
        self._weights = DataFrame(K.eval(self._kernel).T)
