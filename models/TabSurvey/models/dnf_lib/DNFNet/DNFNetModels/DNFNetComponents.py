import tensorflow as tf
import numpy as np


################################################################################################################################################################
# Soft binary gates
################################################################################################################################################################
def and_operator(x, d):
    out = tf.add(x, -d + 1.5)
    out = tf.tanh(out)
    return out


def or_operator(x, d):
    out = tf.add(x, d - 1.5)
    out = tf.tanh(out)
    return out


################################################################################################################################################################
# Localization
################################################################################################################################################################
def broadcast_exp(x, n_formulas, input_dim):
    mu = tf.get_variable('exp_mu', [n_formulas, input_dim], initializer=tf.initializers.random_normal())
    sigma = tf.get_variable('exp_sigma', [1, n_formulas, input_dim], initializer=tf.initializers.random_normal())
    diff = tf.expand_dims(x, axis=1) - tf.expand_dims(mu, axis=0)
    loc = tf.exp(-1 * tf.norm(tf.multiply(diff, sigma), axis=-1))
    return loc


################################################################################################################################################################
# Feature Selection
################################################################################################################################################################
def binary_threshold(x, eps=0.1):
    x = tf.abs(x) - eps
    return 0.5*binary_activation(x) + 0.5


def binary_activation(x):
    forward = tf.sign(x)
    backward = tf.tanh(x)
    return backward + tf.stop_gradient(forward - backward)


def feature_selection(input_dim, keep_feature_prob_arr, n_literals_per_formula_arr, n_formulas, elastic_net_beta):
    binary_threshold_eps = 1
    literals_random_mask,  formulas_random_mask = create_random_mask(input_dim, keep_feature_prob_arr, n_literals_per_formula_arr, n_formulas)
    n_effective_features = np.sum(formulas_random_mask, axis=0)
    formulas_random_mask = tf.constant(formulas_random_mask)
    ext_matrix = extension_matrix(n_formulas, n_literals_per_formula_arr)

    elastic_net_alpha = tf.get_variable('elastic_net_alpha', initializer=tf.constant(value=0.))
    learnable_mask = tf.get_variable('learnable_mask', [input_dim, n_formulas], initializer=tf.initializers.constant(binary_threshold_eps + 0.5))

    learnable_mask_01 = binary_threshold(learnable_mask, eps=binary_threshold_eps)

    l2_square_norm_selected = tf.diag_part(tf.matmul(tf.transpose(tf.square(learnable_mask)), formulas_random_mask))
    l1_norm_selected = tf.diag_part(tf.matmul(tf.transpose(tf.abs(learnable_mask)), formulas_random_mask))

    l2 = tf.abs(tf.div(l2_square_norm_selected, n_effective_features) - elastic_net_beta * binary_threshold_eps ** 2)
    l1 = tf.abs(tf.div(l1_norm_selected, n_effective_features) - elastic_net_beta * binary_threshold_eps)
    elastic_net_reg = tf.reduce_mean((l2 * ((1 - tf.nn.sigmoid(elastic_net_alpha)) / 2) + l1 * tf.nn.sigmoid(elastic_net_alpha)))
    learnable_binary_mask = tf_matmul_sparse_dense(ext_matrix, learnable_mask_01)
    return learnable_binary_mask, literals_random_mask, elastic_net_reg


def create_random_mask(input_dim, keep_feature_prob_arr, n_literals_per_formula_arr, n_formulas):
    literals_random_mask = []
    formulas_random_mask = []
    p_index = 0
    for i in range(n_formulas):
        if i % len(n_literals_per_formula_arr) == 0 and i != 0:
            p_index = (p_index + 1) % len(keep_feature_prob_arr)

        p_i = keep_feature_prob_arr[p_index]
        mask = np.random.choice([1., 0.], input_dim, p=[p_i, 1 - p_i])
        while np.sum(mask) == 0:
            mask = np.random.choice([1., 0.], input_dim, p=[p_i, 1 - p_i])
        n_literals_in_formula = n_literals_per_formula_arr[i % len(n_literals_per_formula_arr)]
        formulas_random_mask.append(np.copy(mask))
        for _ in range(n_literals_in_formula):
            literals_random_mask.append(np.copy(mask))
    return np.array(literals_random_mask).T.astype('float32'), np.array(formulas_random_mask).T.astype('float32')


def extension_matrix(n_formulas, n_literals_per_formula_arr):
    formula_index = 0
    mat = []
    for i in range(n_formulas):
        n_nodes_in_tree = n_literals_per_formula_arr[i % len(n_literals_per_formula_arr)]
        v = np.zeros(n_formulas)
        v[formula_index] = 1.
        for _ in range(n_nodes_in_tree):
            mat.append(v)
        formula_index += 1
    return denseNDArrayToSparseTensor(np.array(mat).T.astype('float32'))


################################################################################################################################################################
# DNF structure
################################################################################################################################################################
def create_conjunctions_indicator_matrix(total_number_of_literals, total_number_of_conjunctions, conjunctions_depth_arr):
    n_different_depth = len(conjunctions_depth_arr)
    n_literals_in_group = np.sum(conjunctions_depth_arr)
    result = []
    for i in range(total_number_of_conjunctions // n_different_depth):
        s = 0
        for d in conjunctions_depth_arr:
            b = np.zeros(total_number_of_literals).astype('bool')
            b[i * n_literals_in_group + s: i * n_literals_in_group + s + d] = True
            s += d
            result.append(b)
    return np.array(result).T


def create_formulas_indicator_matrix(n_formulas, n_conjunctions_arr):
    result = []
    total_number_of_conjunctions = compute_total_number_of_conjunctions(n_formulas, n_conjunctions_arr)

    base = 0
    for i in range(n_formulas):
        n_conjunctions = n_conjunctions_arr[i % len(n_conjunctions_arr)]
        b = np.zeros(total_number_of_conjunctions).astype('bool')
        b[base: base + n_conjunctions] = True
        result.append(b)
        base += n_conjunctions
    return np.array(result).T


def compute_total_number_of_literals(n_formulas, n_conjunctions_arr, conjunctions_depth_arr):
    return (np.sum(n_conjunctions_arr) * np.sum(conjunctions_depth_arr) * n_formulas) // (len(conjunctions_depth_arr) * len(n_conjunctions_arr))


def compute_total_number_of_conjunctions(n_formulas, n_conjunctions_arr):
    return (n_formulas // len(n_conjunctions_arr)) * np.sum(n_conjunctions_arr)


def compute_n_literals_per_formula(n_conjunctions_arr, conjunctions_depth_arr):
    n_literals_per_formula_arr = []
    for n_conjunctions in n_conjunctions_arr:
        n_literals_per_formula_arr.append((n_conjunctions // len(conjunctions_depth_arr)) * np.sum(conjunctions_depth_arr))
    return n_literals_per_formula_arr


################################################################################################################################################################
# General functions
################################################################################################################################################################
def denseNDArrayToSparseTensor(arr):
    if arr.dtype == bool:
        idx = np.where(arr != False)
    else:
        idx = np.where(arr != 0.0)

    if arr.dtype == bool:
        idx = np.vstack(idx).T
        return tf.SparseTensor(idx, np.ones(idx.shape[0]).astype('float32'), arr.shape)
    else:
        return tf.SparseTensor(np.vstack(idx).T, arr[idx], arr.shape)


def tf_matmul_sparse_dense(sparse_A, dense_B):
    return tf.transpose(tf.sparse.matmul(tf.sparse.transpose(sparse_A), tf.transpose(dense_B)))
