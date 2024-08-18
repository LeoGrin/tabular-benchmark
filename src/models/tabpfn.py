from tabpfn import TabPFNClassifier
from mothernet.prediction import MotherNetClassifier, EnsembleMeta
from mothernet.utils import get_mn_model
import numpy as np

class TabPFNStar(TabPFNClassifier):
    def fit(self, X, y):
        if len(X) > 3000:
            indices = np.random.choice(len(X), 3000, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample, y_sample = X, y
        return super().fit(X_sample, y_sample, overwrite_warning=True)


class MotherNetStar(MotherNetClassifier):
    def __init__(self, **kwargs):
        model_string = "mn_d2048_H4096_L2_W32_P512_1_gpu_warm_08_25_2023_21_46_25_epoch_3940_no_optimizer.pickle"
        model_path = get_mn_model(model_string)
        super().__init__(path=model_path, **kwargs)
    def fit(self, X, y):
        if len(X) > 3000:
            indices = np.random.choice(len(X), 3000, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample, y_sample = X, y
        return super().fit(X_sample, y_sample)
    

class EnsembleMotherNetStar(EnsembleMeta):
    def __init__(self, **kwargs):
        base_estimator = MotherNetStar(**kwargs)
        super().__init__(base_estimator=base_estimator)


