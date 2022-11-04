import numpy as np

from npt.datasets.base import BaseDataset


class DebugDataset(BaseDataset):
    """Used for debugging of row interactions.
    Will dynamically overwrite stuff. In each batch.
    But need to set some fake data here such that metadata gets written
    correctly.
    """
    def __init__(self, c):
        super().__init__(fixed_test_set_index=-5)
        self.c = c

    def load(self):
        """Debug dataset. Has four columns.

        The first are copies from one throw of a random dice.
        I.e. the entire column contains the same data.
        Model has to masked out values by reading off dice value from other
        rows. (Only makes sense in semi-supervised).
        The other three of which are just random data we don't care about.
        """

        # Load data from https://www.openml.org/d/4535

        self.N = 20
        self.D = 6

        data = np.zeros((self.N, self.D))
        # populate with all possible choices
        for i in range(self.D):
            data[:18, i] = np.repeat(range(6), 3)

        self.data_table = data.astype('int')

        self.num_target_cols = []
        self.cat_target_cols = [0]

        self.num_features = []
        self.cat_features = list(range(self.D))

        # TODO: add missing entries to sanity check
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.is_data_loaded = True
