import tensorflow as tf
from DNFNet.DNFNetModels.DNFNetComponents import compute_total_number_of_literals, compute_total_number_of_conjunctions


def fetch_placeholders(placeholders):
    x = placeholders['x']
    y = placeholders['y']
    lr = placeholders['lr']
    return x, y, lr


###########################################################################################
# Fully trained FCN
###########################################################################################
class MNN(object):
    def __init__(self, config):
        self.config = config
        self.loaded_input_masks = None

    def build(self, placeholders, n_labels, phase):
        x, y, lr = fetch_placeholders(placeholders)

        n_conjunctions_arr = self.config['n_conjunctions_arr']
        conjunctions_depth_arr = self.config['conjunctions_depth_arr']
        n_formulas = self.config['n_formulas']
        total_number_of_literals = compute_total_number_of_literals(n_formulas, n_conjunctions_arr, conjunctions_depth_arr)
        total_number_of_conjunctions = compute_total_number_of_conjunctions(n_formulas, n_conjunctions_arr)

        out = tf.layers.dense(x, total_number_of_literals, activation=tf.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
        out = tf.layers.dense(out, total_number_of_conjunctions, activation=tf.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
        out = tf.layers.dense(out, n_formulas, activation=tf.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
        output = tf.layers.dense(out, n_labels, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope('loss'):
            if n_labels > 1:
                loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)
                output = tf.nn.softmax(output)
            else:
                loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=output)
                output = tf.nn.sigmoid(output)

            loss += tf.losses.get_regularization_loss()

            if phase == 'train':
                print('Build optimizer')
                optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
            else:
                optimizer = None

            build_dict = {'optimizer': optimizer, 'loss': loss, 'output': output, 'features': None, 'input_masks': None}
            return build_dict

