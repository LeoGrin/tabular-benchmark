import wandb
import numpy as np

sweep_config = {
  "program": "run_experiment.py",
  "name" : "hgbt_categorical_regression_default",
  "project": "thesis-4",
  "method" : "grid",
  "metric": {
    "name": "mean_test_score",
    "goal": "minimize"
  },
  "parameters" : {
    "model_type": {
      "value": "sklearn"
    },
    "model_name": {
      "value": "hgbt_r"
    },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["yprop_4_1",
                  "analcatdata_supreme",
                  "visualizing_soil",
                  "black_friday",
                  "nyc-taxi-green-dec-2016",
                  "diamonds",
                  "Allstate_Claims_Severity",
                  "Mercedes_Benz_Greener_Manufacturing",
                  "Brazilian_houses",
                  "Bike_Sharing_Demand",
                  "OnlineNewsPopularity",
                  "house_sales",
                  "LoanDefaultPrediction",
                  "particulate-matter-ukair-2017",
                  "SGEMM_GPU_kernel_performance"]
    },
    "n_iter": {
      "value": "auto",
    },
    "regression": {
          "value": True
    },
    "data__regression": {
          "value": True
    },
    "data__categorical": {
      "value": True
    },
    "transformed_target": {
      "values": [False]
    },
    "max_train_samples": {
      "value": 10000
    },
  }
}


sweep_id = wandb.sweep(sweep_config, project="thesis-4")