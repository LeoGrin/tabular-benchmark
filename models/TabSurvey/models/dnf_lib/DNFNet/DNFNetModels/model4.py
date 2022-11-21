import numpy as np
import tensorflow as tf
from DNFNet.DNFNetModels.DNFNetComponents import compute_total_number_of_literals, compute_total_number_of_conjunctions, compute_n_literals_per_formula,\
    denseNDArrayToSparseTensor, create_conjunctions_indicator_matrix, create_formulas_indicator_matrix, feature_selection, and_operator, or_operator, tf_matmul_sparse_dense


def fetch_placeholders(placeholders):
    x = placeholders['x']
    y = placeholders['y']
    lr = placeholders['lr']
    return x, y, lr


###########################################################################################
# DNF structure with feature selection
###########################################################################################
class MNN(object):
    def __init__(self, config):
        self.config = config
        self.loaded_input_masks = None

    def build(self, placeholders, n_labels, phase):
        x, y, lr = fetch_placeholders(placeholders)

        input_dim = self.config['input_dim']
        n_conjunctions_arr = self.config['n_conjunctions_arr']
        conjunctions_depth_arr = self.config['conjunctions_depth_arr']
        n_formulas = self.config['n_formulas']

        total_number_of_literals = compute_total_number_of_literals(n_formulas, n_conjunctions_arr, conjunctions_depth_arr)
        total_number_of_conjunctions = compute_total_number_of_conjunctions(n_formulas, n_conjunctions_arr)
        n_literals_per_formula_arr = compute_n_literals_per_formula(n_conjunctions_arr, conjunctions_depth_arr)
        conjunctions_indicator_matrix = create_conjunctions_indicator_matrix(total_number_of_literals, total_number_of_conjunctions, conjunctions_depth_arr)
        formulas_indicator_matrix = create_formulas_indicator_matrix(n_formulas, n_conjunctions_arr)

        conjunctions_indicator_sparse_matrix = denseNDArrayToSparseTensor(conjunctions_indicator_matrix)
        formulas_indicator_sparse_matrix = denseNDArrayToSparseTensor(formulas_indicator_matrix)
        and_bias = np.sum(conjunctions_indicator_matrix, axis=0)
        or_bias = np.sum(formulas_indicator_matrix, axis=0)
        learnable_binary_mask, literals_random_mask, elastic_net_reg = feature_selection(input_dim, self.config['keep_feature_prob_arr'], n_literals_per_formula_arr, n_formulas, self.config['elastic_net_beta'])

        bias = tf.get_variable('literals_bias', [total_number_of_literals], initializer=tf.zeros_initializer())
        weight = tf.get_variable('literals_weights', [input_dim, total_number_of_literals], initializer=tf.contrib.layers.xavier_initializer())
        weight = tf.multiply(weight, tf.constant(literals_random_mask))

        out_literals = tf.tanh(tf.add(tf.matmul(x, tf.multiply(weight, learnable_binary_mask)), bias))
        out_conjunctions = and_operator(tf_matmul_sparse_dense(conjunctions_indicator_sparse_matrix, out_literals), d=and_bias)
        out_DNNFs = or_operator(tf_matmul_sparse_dense(formulas_indicator_sparse_matrix, out_conjunctions), d=or_bias)
        out_DNNFs = tf.reshape(out_DNNFs, shape=(-1, n_formulas))

        output = tf.layers.dense(out_DNNFs, n_labels, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope('loss'):
            if n_labels > 1:
                loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)
                output = tf.nn.softmax(output)
            else:
                loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=output)
                output = tf.nn.sigmoid(output)

            loss += elastic_net_reg
            loss += tf.losses.get_regularization_loss()

            if phase == 'train':
                print('Build optimizer')
                optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
            else:
                optimizer = None

            build_dict = {'optimizer': optimizer, 'loss': loss, 'output': output, 'features': None, 'input_masks': None}
            return build_dict

