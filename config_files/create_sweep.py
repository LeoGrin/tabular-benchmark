import wandb

sweep_config = {
  "program": "run_experiment.py",
  "name" : "first_sweep_rf",
  "method" : "random",
  "metric": {
    "name": "test_score",
    "goal": "maximize"
  },
  "parameters" : {
    "model_type": {
      "value": "sklearn"
    },
    "model_name": {
      "value": "rf_c"
    },
    "model__n_estimators": {
      "values": [1, 2, 5, 10, 100, 200, 300, 400, 500]
    },
    "model__min_weight_fraction_leaf":{
      "min": 0.0,
      "max": 0.5
    },
    "data__method_name": {
      "value": "real_data"
    },
    "data__keyword": {
      "values": ["heloc", "electricity", "california", "covtype", "spam", "churn", "credit", "shopping", "nomao", "cpu", "wine"]
    },
    "max_train_samples": {
      "value": 10000
    },
    "train_set_prop": {
      "value": 0.75
    },
    "regression": {
      "value": False
    }
  }
}


sweep_id = wandb.sweep(sweep_config)