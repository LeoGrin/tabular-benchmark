import random
from utils.tree import Forest, Tree
import numpy as np
from utils.utils import softmax
import matplotlib.pyplot as plt
import torch.nn.functional as F

def generate_random_tree(x, n_classes, depth, split_distribution="uniform", split_param=1, rng=None):
    """
    Generate a random tree which labels the data.
    :param x: data to label
    :param n_classes: number of classes to choose for the labels
    :param depth: depth of the random tree
    :param split_distribution:{"uniform", "gaussian"}, default="uniform"
     Distribution from which is sampled the split threshold at each step.
     If "uniform": Uniform(-split_param * feature_std, +split_param * feature_std)
     If "gaussian": N(0, split_param * feature_std)
    :param split_param: parameter controlling the spread of the split threshold distribution
    WARNING: not implement yet for "uniform" split distribution
    :return: a Tree object with a fit and predict methods
    """

    def generate_tree(x, n_classes, depth, parent=None, prediction=None, min_num_leaf=5):
        #TODO allow random stopping of a substree?
        if x.shape[0] < min_num_leaf or depth == 1:
            #TODO: when stopping early, correct the depth of all parents
            #you should look at all leaf of a tree for this
            assert depth > 1 or prediction is not None
            if depth > 1:
                prediction = rng.choice(range(n_classes))
            leaf = Tree(depth, parent)
            leaf.set_prediction(prediction)
            return leaf
        else:
            x_median = np.quantile(x, 0.5, axis=0)
            x_25 = np.quantile(x, 0.25, axis=0)
            x_75 = np.quantile(x, 0.75, axis=0)
            n_features = x.shape[1]

            tree = Tree(depth, parent)
            split_feature = rng.choice(range(n_features), 1)[0]
            # we want to sample a split threshold depending on the variance of this feature in our data
            if split_distribution == "uniform":
                #TODO allow split_param
                split_threshold = rng.uniform(x_25[split_feature],
                                                    x_75[split_feature])
            if split_distribution == "gaussian":
                split_threshold = rng.normal(loc=x_median[split_feature], scale=split_param * (x_75[split_feature] - x_25[split_feature]))

            tree.set_split(split_feature, split_threshold)
            if depth == 2: #make sure two adjacent leaves have different predictions
                prediction_left = rng.choice(range(n_classes))
                prediction_right = (prediction_left + 1) % 2
            else:
                prediction_left = None
                prediction_right = None
            tree.right = generate_tree(x[x[:, split_feature] >= split_threshold],
                                       n_classes,
                                       depth - 1,
                                       tree, prediction_left)
            tree.left = generate_tree(x[x[:, split_feature] < split_threshold],
                                      n_classes,
                                      depth - 1,
                                      tree,
                                      prediction_right)
            return tree

    root = generate_tree(x, n_classes, depth, parent=None)
    return root


def generate_random_forest(x, n_classes=2, n_trees=5, max_depth=5, depth_distribution="constant",
                                       split_distribution="uniform", split_param=1, rng=None):
    """

    :param x: data to label
    :param n_classes: number of classes to choose for the labels
    :param n_trees: number of trees in the forest
    :param max_depth: depth of the random tree
    :param depth_distribution:{"constant", "uniform"} Distribution from which are sampled the tree depths
    if "constant", every depths are max_depth
    if "uniform", uniform in [2, ..., max_depth]
    :param split_distribution:{"uniform", "gaussian"}, default="uniform"
     Distribution from which is sampled the split threshold at each step.
     If "uniform": Uniform(-split_param * feature_std, +split_param * feature_std)
     If "gaussian": N(0, split_param * feature_std)
    :param split_param: parameter controlling the spread of the split threshold distribution
    :return: a Forest object with predict and fit methods
    """
    if depth_distribution == "constant":
        depths = [max_depth] * n_trees
    elif depth_distribution == "uniform":
        depths = [rng.randint(2, max_depth + 1) for i in range(n_trees)]
    #print("depths {}".format(depths))
    trees = [generate_random_tree(x, n_classes, depths[i], split_distribution, split_param, rng) for i in range(n_trees)]
    forest = Forest(trees, rng)

    return forest

def generate_labels_random_forest(x, n_classes=2, n_trees=5, max_depth=5, depth_distribution="constant",
                                       split_distribution="uniform", split_param=1, rng=None):
    forest = generate_random_forest(x, n_classes, n_trees, max_depth, depth_distribution,
                                    split_distribution, split_param, rng)

    return forest.predict(x)

def generate_labels_sparse_in_interaction(x, n_interactions=10, ensemble_size=2, variant="sum", rng=None):
    """
    :param x:
    :param n_interactions: number of interactions to generate
    :param ensemble_size: number of features interacting in each interaction
    :param variant: see below
    :return: y
    """
    #assume x is N(0, 1)
    if type(ensemble_size) == float or type(ensemble_size) == np.float64 or type(ensemble_size) == np.float32:
        ensemble_size = int(ensemble_size * x.shape[1])
    relu = lambda x: np.maximum(0, x)
    function_list = [np.sin,
                     np.cos,
                     relu,
                     lambda x:np.log(relu(x) + 0.2)]
    y = np.zeros(x.shape[0])
    col_idx = list(range(x.shape[1]))
    for i in range(n_interactions):
        if variant == "sum":
            #apply a random function to the sum of the chosen features
            idxs = rng.choice(col_idx, ensemble_size, replace=False)
            function = rng.choice(function_list, 1)[0]
            y += np.array(list(map(function, np.sum(x[:, idxs], axis=1))))
        elif variant == "transform_sum":
            #apply a random function to each (randomly chosen) feature, then to the sum of these features
            idxs = rng.choice(col_idx, ensemble_size, replace=False)
            new_x = x[:, idxs]
            for j in range(new_x.shape[1]):
                function = rng.choice(function_list, 1)[0]
                new_x[:, j] = np.array(list(map(function, new_x[:, j])))
            function = rng.choice(function_list, 1)[0]
            y += np.array(list(map(function, np.sum(new_x, axis=1))))
        elif variant == "hierarchical":
            #sum pairs of columns and transforms them until we have shape 1
            idxs = rng.choice(col_idx, ensemble_size, replace=False)
            new_x = x[:, idxs]
            for j in range(new_x.shape[1]):
                function = rng.choice(function_list, 1)[0]
                new_x[:, j] = np.array(list(map(function, new_x[:, j])))
            while new_x.shape[1] >= 2:
                function = rng.choice(function_list, 1)[0]
                idxs = rng.choice(range(new_x.shape[1]), 2, replace=False)
                new_x = np.concatenate([np.delete(new_x, idxs, 1),
                                        np.array(list(map(function, np.sum(new_x[:, idxs], axis=1)))).reshape(-1, 1)],
                                       axis=1)
            function = rng.choice(function_list, 1)[0]
            y += np.array(list(map(function, np.sum(new_x, axis=1))))

    return y > np.median(y)



def generate_labels_linear(x, noise_level=0.5, weights="equal", rng=None):
    """
    :param x: data to label
    :param noise_level: control the size of the noise
    :param weights:{"constant", "random"} Control how the weights of the linear combination are generated
    if "equal", each feature has the same weight
    if "random", the weights are a random vector summing to 1
    :return: array of labels
    """
    #TODO class imbalance
    n_samples, n_features = x.shape
    #generate random weights
    if weights == "random":
        w = rng.normal(0, 1, x.shape[1])
    elif weights == "equal":
        w = np.ones(x.shape[1])
    w = softmax(w)
    y = x @ w
    noise = rng.normal(0, np.std(y) * noise_level)
    y += noise
    labels = (y > np.median(y)).astype(np.int32)
    return labels

def generate_labels_xor_2d(x):
    assert x.shape[1] == 2
    medians = np.median(x, axis=0)
    y = np.logical_xor(x[:, 0] > medians[0], x[:, 1] > medians[1]).astype(np.int32)
    return y

def last_column_as_target(x):
    y = x[:,-1]
    return y > np.median(y)

def periodic_sinus(x, period=None, offset=None, period_size=None, noise=True, rng=None):
    #offset = offset * 2
    if not period_size is None:
        if not period is None:
            offset = (4 - period_size * period) / 2
        elif not offset is None:
            period = (4 - offset * 2) / period_size
    res = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if x[i] < -2 + offset or x[i] > 2 - offset:
            res[i] = 0
        else:
            res[i] = np.sin(x[i] * (2 * np.pi) / (4 - 2 * offset) * period)
    #res = np.sin(x.reshape(-1) * np.pi / period).astype(np.float32)
    if noise:
        res += rng.normal(0, 0.1, x.shape[0])
    return res

def periodic_triangle(x, n_periods=None, offset=None, period_size=None, noise=True, rng=None):
    # TAKE INTO INPUT A UNIFORM(-2, 2) (I think)
    assert (x <= 2).all()
    assert (x >= -2).all()
    #offset = offset * 2
    if not period_size is None:
        if not n_periods is None:
            offset = (4 - period_size * n_periods) / 2
        elif not offset is None:
            period = (4 - offset * 2) / period_size
    res = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if x[i] < -2 + offset or x[i] > 2 - offset:
            res[i] = 0
        else:
            res[i] = 2 * np.abs(np.abs(x[i]) / period_size - int(np.abs(x[i]) / period_size + 1/2))
            #res[i] = np.sin(x[i] * (2 * np.pi) / (4 - 2 * offset) * period)
    #res = np.sin(x.reshape(-1) * np.pi / period).astype(np.float32)
    if noise:
        res += rng.normal(0, 0.1, x.shape[0])
    return res



if __name__ == """__main__""":
    #forest = generate_random_forest(np.ones((10, 10)), 2)

    x = np.random.normal(size=(1000, 10))
    #labels = generate_labels_linear(x)
    labels = generate_labels_sparse_in_interaction(x)
    plt.scatter(x[:, 0], x[:, 1], c=labels)
    plt.show()
