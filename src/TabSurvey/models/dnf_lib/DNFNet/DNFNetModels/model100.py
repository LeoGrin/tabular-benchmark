import tensorflow as tf


def fetch_placeholders(placeholders):
    x = placeholders['x']
    y = placeholders['y']
    lr = placeholders['lr']
    dropout_rate = placeholders['dropout_rate']
    is_training = placeholders['is_training']
    return x, y, lr, dropout_rate, is_training


###########################################################################################
# FCN
###########################################################################################
class MNN(object):
    def __init__(self, config):
        self.config = config
        self.loaded_input_masks = None

    def build(self, placeholders, n_labels, phase):
        x, y, lr, _, is_training = fetch_placeholders(placeholders)

        net_description = '#'*50 + '\n'
        out = x
        for layer_width in self.config['FCN_layers']:
            in_shape = out.get_shape()[1].value
            net_description += '{} -> {} \n'.format(in_shape, layer_width)

            out = tf.layers.dense(out, layer_width, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.config['FCN_l2_lambda']), kernel_initializer=tf.contrib.layers.xavier_initializer())
            out = tf.layers.dropout(inputs=out, rate=self.config['dropout_rate'], training=is_training)

        in_shape = out.get_shape()[1].value
        net_description += '{} -> {} \n'.format(in_shape, n_labels)
        net_description += '#' * 50 + '\n'
        print(net_description)

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

