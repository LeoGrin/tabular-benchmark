import numpy as np

class Tree:
    def __init__(self, depth, parent=None):
        self.parent = parent
        self.left = None
        self.right = None
        self.root_feature = None
        self.root_threshold = None
        self.depth = depth
        self.predicion = None

    def set_split(self, feature_id, threshold):
        self.root_threshold = threshold
        self.root_feature = feature_id

    def set_prediction(self, prediction):
        assert self.left is None and self.right is None  # leaf
        self.predicion = prediction

    def predict(self, x):
        #x shape (n_features,)
        if not self.predicion is None:
            return self.predicion
        else:
            if x[self.root_feature] > self.root_threshold:
                return self.right.predict(x)
            else:
                return self.left.predict(x)


class Forest:
    def __init__(self, tree_list, rng):
        self.tree_list = tree_list
        self.rng = rng
    def predict(self, x):
        #x shape (n_samples, n_features)

        # for tree in self.tree_list:
        #     print("depth: {}".format(tree.depth))
        #     print("split {}".format(tree.root_threshold))#
        #
        #     sum = 0
        #     for sample in x:
        #         sum += tree.predict(sample)
        #     print(sum)
        predictions = []
        for sample in x: #TODO vectorize ?
            values, counts = np.unique([tree.predict(sample) for tree in self.tree_list], return_counts=True)
            indices_max = np.argwhere(counts == np.amax(counts)).flatten()
            prediction = self.rng.choice(values[indices_max], 1)[0]
            predictions.append(prediction)
        return np.array(predictions)