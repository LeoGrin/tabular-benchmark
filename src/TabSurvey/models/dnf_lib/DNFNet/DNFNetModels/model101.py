import tensorflow as tf
from DNFNet.DNFNetModels.DNFNetComponents import binary_threshold


def fetch_placeholders(placeholders):
    x = placeholders['x']
    y = placeholders['y']
    lr = placeholders['lr']
    return x, y, lr


###########################################################################################
# FCN - syn experiment
###########################################################################################
class MNN(object):
    def __init__(self, config):
        self.config = config

    def build(self, placeholders, n_labels, phase):
        x, y, lr = fetch_placeholders(placeholders)
        mask = self.config['mask']
        elastic_net_reg = None

        if self.config['model_type'] == 'FCN_with_oracle_mask':
            m = tf.constant(mask, dtype='float32')
            x_i = tf.multiply(x, m)
        elif self.config['model_type'] == 'FCN_with_feature_selection':
            binary_threshold_eps = 1.
            feature_selector = tf.get_variable('feature_selector', [self.config['input_dim']], initializer=tf.initializers.constant(binary_threshold_eps + 0.5))
            feature_selector_01 = binary_threshold(feature_selector, eps=binary_threshold_eps)
            x_i = tf.multiply(feature_selector_01, x)

            elastic_net_alpha = tf.get_variable('elastic_net_alpha', initializer=tf.constant(value=0.))
            l2 = tf.abs(tf.div(tf.square(tf.norm(feature_selector, ord=2)), self.config['input_dim']) - self.config['elastic_net_beta'] * binary_threshold_eps**2)
            l1 = tf.abs(tf.div(tf.norm(feature_selector, ord=1), self.config['input_dim']) - self.config['elastic_net_beta'] * binary_threshold_eps)
            elastic_net_reg = l2 * ((1 - tf.nn.sigmoid(elastic_net_alpha)) / 2) + l1 * tf.nn.sigmoid(elastic_net_alpha)
        elif self.config['model_type'] == 'FCN':
            x_i = x
        else:
            raise Exception('ERROR: in model_type')

        out = tf.layers.dense(x_i, 64, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        out = tf.layers.dense(out, 32, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        output = tf.layers.dense(out, n_labels, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope('loss'):
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=output)
            output = tf.nn.sigmoid(output)
            loss += tf.losses.get_regularization_loss()

            if elastic_net_reg is not None:
                loss += elastic_net_reg

            if phase == 'train':
                print('Build optimizer')
                optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
            else:
                optimizer = None

            build_dict = {'optimizer': optimizer, 'loss': loss, 'output': output, 'features': None}
            return build_dict

