import os
os.environ["PROJECT_DIR"] = "test"
from generate_dataset_pipeline import generate_dataset
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils.plot_utils import plot_decision_boudaries
import matplotlib.pyplot as plt


# config = {"model_name": "rf_c",
#           "model_type": "sklearn",
#           "data__method_name": "real_data",
#           "data__keyword": "california",
#           "transform__0__method_name": "gaussienize",
#           "transform__0__type": "quantile",
#           #"transform__1__method_name": "select_features_rf",
#           #"transform__1__num_features": 2,
#           "transform__1__method_name": "remove_high_frequency_from_train",
#           "transform__1__cov_mult": 2,
#           "train_prop": 0.70,
#           "max_train_samples": 10000,
#           "val_test_prop": 0.3,
#           "max_val_samples": 50000,
#          "max_test_samples": 50000}

config = {"model_name": "rtdl_resnet",
          "model_type": "skorch",
          #"model__n_estimators": 2000,
          "data__method_name": "real_data",
          "data__categorical": True,
          "data__keyword": "compass",
          # "transform__0__method_name": "gaussienize",
          # "transform__0__type": "quantile",
          # "transform__1__method_name": "select_features_rf",
          # "transform__1__num_features": 5,
          # "transform__2__method_name": "remove_high_frequency_from_train",
          # "transform__2__cov_mult": 0.5,
          # "transform__2__covariance_estimation": "robust",
          "train_prop": 0.70,
          "max_train_samples": 10000,
          "val_test_prop": 0.3,
          "max_val_samples": 10000,
         "max_test_samples": 10000}

rng = np.random.RandomState(32)
x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator = generate_dataset(config, rng)

print(x_train.shape)

rf = RandomForestClassifier()
print(x_train.shape)
#print(x_train[:, :2].shape)
rf.fit(x_train, y_train)
print(rf.score(x_train, y_train))
print(rf.score(x_val, y_val))
print(rf.score(x_test, y_test))
print("########")
rf.fit(x_train[:, :2], y_train)
print(rf.score(x_train[:, :2], y_train))
print(rf.score(x_val[:, :2], y_val))
print(rf.score(x_test[:, :2], y_test))
plot_decision_boudaries(x_train[:, :2], y_train, x_test[:, :2], y_test, rf,
                       x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5)
plt.show()
plot_decision_boudaries(x_train[:, :2], y_train, x_test[:, :2], y_test, rf,
                       x_min=-2, x_max=2, y_min=-2, y_max=2)
#plt.title("cov_mult = {}".format(config["transform__1__cov_mult"]))
plt.show()



