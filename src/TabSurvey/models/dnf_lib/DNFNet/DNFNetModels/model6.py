import tensorflow as tf
from DNFNet.DNFNetModels.DNFNetComponents import compute_total_number_of_literals, compute_total_number_of_conjunctions, compute_n_literals_per_formula, \
    feature_selection, broadcast_exp


def fetch_placeholders(placeholders):
    x = placeholders['x']
    y = placeholders['y']
    lr = placeholders['lr']
    return x, y, lr


###########################################################################################
# FCN with feature selection and localization
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

        learnable_binary_mask, literals_random_mask, elastic_net_reg = feature_selection(input_dim, self.config['keep_feature_prob_arr'], n_literals_per_formula_arr, n_formulas, self.config['elastic_net_beta'])

        bias = tf.get_variable('literals_bias', [total_number_of_literals], initializer=tf.zeros_initializer())
        weight = tf.get_variable('literals_weights', [input_dim, total_number_of_literals], initializer=tf.contrib.layers.xavier_initializer())
        weight = tf.multiply(weight, tf.constant(literals_random_mask))

        out = tf.tanh(tf.add(tf.matmul(x, tf.multiply(weight, learnable_binary_mask)), bias))
        out = tf.layers.dense(out, total_number_of_conjunctions, activation=tf.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
        out = tf.layers.dense(out, n_formulas, activation=tf.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())

        loc = broadcast_exp(x, n_formulas, self.config['input_dim'])
        temperature = tf.get_variable('temperature', initializer=tf.constant(value=2.))
        loc = tf.nn.softmax(tf.sigmoid(temperature) * loc)
        out = tf.multiply(out, loc)

        output = tf.layers.dense(out, n_labels, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

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

