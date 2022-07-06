import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "hgbt_categorical_classif_large",
  "project": "thesis-4",
  "method" : "random",
  "metric": {
    "name": "mean_test_score",
    "goal": "minimize"
  },
  "parameters" : {
    "model_type": {
      "value": "sklearn"
    },
    "model_name": {
      "value": "hgbt_c"
    },
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    #"model__loss": {
    #  "values": ["squared_error", "absolute_error"],
    #},
    "model__learning_rate": {
      'distribution': "log_normal",
      'mu': float(np.log(0.01)),
      'sigma': float(np.log(10.0)),
    },
    "model__max_leaf_nodes": {
        'distribution': "q_normal",
        'mu': 31,
        "sigma": 5
    },
    "model__max_depth": { # Added None compared to hyperopt
      "values": [None, 2, 3, 4],
      "probabilities": [0.1, 0.1, 0.7, 0.1]
    },
    "model__min_samples_leaf": { # Not exactly like Hyperopt
      "distribution": "q_normal",
      "mu": 20,
      "sigma": 2,
      "q": 1
    },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["electricity",
                 #"eye_movements",
                 # "KDDCup09_upselling",
                  "covertype",
                 # "rl",
                  "road-safety"]
                  #"compass"]
    },
    "n_iter": {
      "value": "auto",
    },
    "regression": {
          "value": False
    },
    "data__regression": {
          "value": False
    },
    "data__categorical": {
            "value": True
    },
    "max_train_samples": {
      "value": 50000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-4")