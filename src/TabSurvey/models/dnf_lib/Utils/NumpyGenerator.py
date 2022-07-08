import numpy as np
import tensorflow as tf


class NumpyGenerator(object):
    def __init__(self, x, y, output_dim, batch_size, scale=None, translate_label_to_one_hot=True, shuffle=True, copy_dataset=True):
        self.batch_size = batch_size
        self.scale = scale
        self.output_dim = output_dim
        self.shuffle = shuffle
        self.translate_label_to_one_hot = translate_label_to_one_hot
        self.index = 0
        self.epochs = 0
        self.dataset_size = x.shape[0]
        self.indices = np.arange(x.shape[0])

        if copy_dataset:
            self.x = np.copy(x)
            self.y = np.copy(y)
        else:
            self.x = x
            self.y = y

        if self.shuffle:
            self._shuffle_dataset()

    def _get_values_by_range(self, begin, end):
        batch_ix = self.indices[begin: end]
        x_batch = self.x[batch_ix].copy()
        y_batch = self.y[batch_ix].copy()

        if self.translate_label_to_one_hot:
            y_batch = tf.keras.utils.to_categorical(y_batch, num_classes=self.output_dim)
        elif self.output_dim == 1:
            y_batch = np.expand_dims(y_batch, axis=1)

        if self.scale is not None:
            x_batch = self.scale(x_batch)

        x_batch = x_batch.astype('float32')
        return x_batch, y_batch

    def _shuffle_dataset(self):
        np.random.shuffle(self.indices)

    def _build_feed_dict(self, x, y):
        feed_dict = dict()
        feed_dict['x'] = x
        feed_dict['y'] = y
        return feed_dict

    def reset(self):
        self.index = 0
        self.epochs = 0

    def get_epochs(self):
        return self.epochs

    def get_dataset_size(self):
        return self.dataset_size

    def get_batch(self):
        batch_size = self.batch_size
        if self.index + batch_size < self.dataset_size:
            batch_features, batch_labels = self._get_values_by_range(self.index, self.index + batch_size)
            self.index += batch_size
        else:
            batch_features, batch_labels = self._get_values_by_range(self.index, self.dataset_size)
            self.index = 0
            self.epochs += 1
            if self.shuffle:
                self._shuffle_dataset()
        return self._build_feed_dict(batch_features, batch_labels)