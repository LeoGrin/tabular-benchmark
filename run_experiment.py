from generate_dataset_pipeline import generate_dataset
import sys, traceback  # Needed for pulling out your full stackframe info
from train import *
from create_models import create_model

def train_model_on_config(config):
    try:
        rng = np.random.RandomState(0)  #FIXME
        x_train, x_test, y_train, y_test = generate_dataset(config, rng) #FIXME

        model = create_model(config)

        model = train_model(model, x_train, y_train, config)

        train_score, test_score = evaluate_model(model, x_train, y_train, x_test, y_test, config)
        print(train_score)
        print(test_score)

    except:
        # Print to the console
        print("ERROR")
        # To get the traceback information
        print(traceback.format_exc())
        print(config)
        return {}

    return {}

if __name__ ==  """__main__""":
    config = {"model_type": "sklearn",
              "model_name": "rf_r",
              "model__n_estimators": 1,
              "data__n_samples": 300,
              "data__n_features": 1,
              "data__method_name": "uniform_data",
              "target__method_name": "periodic_triangle",
              "target__n_periods": 8,
              "target__period_size": 0.2,
              "max_train_samples": None,
              "regression": True}

    train_model_on_config(config)