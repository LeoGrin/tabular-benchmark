import torch

from npt.datasets.base import BaseDataset


class CIFAR10Dataset(BaseDataset):
    def __init__(self, c):
        super().__init__(
            fixed_test_set_index=None)
        self.c = c

    def load(self):
        """
        Classification dataset.

        Target in last column.
        60 000 rows.
        3072 attributes.
        1 class (10 class labels)

        Author: Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
        Source: [University of Toronto]
        (https://www.cs.toronto.edu/~kriz/cifar.html) - 2009
        Alex Krizhevsky (2009) Learning Multiple Layers of Features from
            Tiny Images, Tech Report.

        CIFAR-10 is a labeled subset of the [80 million tiny images dataset]
            (http://groups.csail.mit.edu/vision/TinyImages/).

        It (originally) consists 32x32 color images representing
            10 classes of objects:
                0. airplane
                1. automobile
                2. bird
                3. cat
                4. deer
                5. dog
                6. frog
                7. horse
                8. ship
                9. truck

        CIFAR-10 contains 6000 images per class.
        Similar to the original train-test split, which randomly divided
            these classes into 5000 train and 1000 test images per class,
             we do 5-fold class-balanced cross-validation by default.

        The classes are completely mutually exclusive.
        There is no overlap between automobiles and trucks.
        "Automobile" includes sedans, SUVs, things of that sort.
        "Truck" includes only big trucks. Neither includes pickup trucks.

        ### Attribute description
        Each instance represents a 32x32 colour image as a 3072-value array.
        The first 1024 entries contain the red channel values, the next
        1024 the green, and the final 1024 the blue. The image is stored
        in row-major order, so that the first 32 entries of the array are
        the red channel values of the first row of the image.

        The labels are encoded as integers in the range 0-9,
            corresponding to the numbered classes listed above.
        """
        self.N = 60000
        self.D = 3073
        self.cat_features = [self.D - 1]
        self.num_features = list(range(0, self.D - 1))

        # Target col is the last feature
        self.num_target_cols = []
        self.cat_target_cols = [self.D - 1]

        # TODO: add missing entries to sanity check
        self.missing_matrix = torch.zeros((self.N, self.D), dtype=torch.bool)
        self.is_data_loaded = True

        self.input_feature_dims = [1] * 3072
        self.input_feature_dims += [10]
