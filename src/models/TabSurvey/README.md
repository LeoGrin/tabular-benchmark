# TabSurvey

Basis for various experiments on deep learning models for tabular data.
See the [Deep Neural Networks and Tabular Data: A Survey](https://arxiv.org/abs/2110.01889) paper.

## How to use

### Using the docker container

The code is designed to run inside a docker container. See the `Dockerfile`.
In the docker file, different conda environments are specified for the various 
requirements of the models. Therefore, building the container for the first time takes a
while.

Just build it as usual via `docker build -t <image name> <path to Dockerfile>`.

To start the docker container then run:

``docker run -v ~/output:/opt/notebooks/output -p 3123:3123 --rm -it --gpus all <image name>``

- The `-v ~/output:/opt/notebooks/output` option is recommended to have access to the 
outputs of the experiments on your local machine.

- The `docker run` command starts a jupyter notebook (to have a nice editor for small changes or experiments).
To have access to the notebook from outside the docker container, `-p 3123:3123` connects the notebook to your local 
machine. You can change the port number in the `Dockerfile`.

- If you have GPUs available, add also the `--gpus all` option to have access to them from
inside the docker container.

To enter the running docker container via the command do the following:
- Call `docker ps` to find the ID of the running container.
- Call `docker exec -it <container id> bash` to enter the container. 
Now you can navigate to the right directory with `cd opt/notebooks/`.

----------------------------

### Run a single model on a single dataset

To run a single model on a single dataset call:

``python train.py --config/<config-file of the dataset>.yml --model_name <Name of the Model>``

All parameters set in the config file, can be overwritten by command line arguments, for example:

- ``--optimize_hyperparameters`` Uses [Optuna](https://optuna.org/) to run a hyperparameter optimization. If not set, the parameters listed in the `best_params.yml` file are used.

- ``--n_trails <number trials>`` Number of trials to run for the hyperparameter search

- ``--epochs <number epochs>`` Max number of epochs

- ``--use_gpu`` If set, available GPUs are used (specified by `gpu_ids`)

- ... and so on. All possible parameters can be found in the config files or calling: 
``python train.y -h``

If you are using the docker container, first enter the right conda environment using `conda activate <env name>` to 
have all required packages. The `train.py` file is in the `opt/notebooks/` directory.

--------------------------------------

### Run multiple models on multiple datasets

To run multiple models on multiple datasets, there is the bash script `testall.sh` provided.
In the bash script the models and datasets can be specified. Every model needs to know in 
which conda environment in has to be executed.

If you run inside our docker container, just comment out all models and datasets you don't
want to run and then call:

`./testall.sh`

-------------------------------------
### Computing model attributions (currently supported for SAINT, TabTransformer, TabNet)

The framework provides implementations to compute feature attribution explanations for several models.
Additionally, the feature attributions can be automatically compared to SHAP values and a global ablation 
test which successively perturbs the most important features, can be run. The same parameters as before can be passed, but
with some additions:

`attribute.py --model_name <Name of the Model> [--globalbenchmark] [--compareshap] [--numruns <int>] [--strategy diag]`

- `--globalbenchmark` Additionally run the global perturbation benchmark

- `--compareshap` Compare attributions to shapley values

- `--numruns <number run>` Number of repetitions for the global benchmark

- ``--strategy diag`` SAINT and TabTransformer support another attribution strategy, where the diagonal of the attention map is used. Pass this argument to use it.


-------------------------------------

## Add new models

Every new model should inherit from the base class `BaseModel`. Implement the following methods:

- `def __init__(self, params, args)`: Define your model here.
- `def fit(self, X, y, X_val=None, y_val=None)`: Implement the training process. (Return the loss and validation history)
- `def predict(self, X)`: Save and return the predictions on the test data - the regression values or the concrete classes for classification tasks
- `def predict_proba(self, X)`: Only for classification tasks. Save and return the probability distribution over the classes.
- `def define_trial_parameters(cls, trial, args)`: Define the hyperparameters that should be optimized.
- (optional) `def save_model`: If you want to save your model in a specific manner, override this function to.

Add your `<model>.py` file to the `models` directory and do not forget to update the `models/__init__.py` file.

----------------------------------------------

## Add new datasets

Every dataset needs a config file specifying its features. Add the config file to the `config` directory.

Necessary information are:
- *dataset*: Name of the dataset
- *objective*: Binary, classification or regression task
- *direction*: Direction of optimization. In the current implementation the binary scorer returns the AUC-score,
hence, should be maximized. The classification scorer uses the log loss and the regression scorer mse, therefore
both should be minimized.
- *num_features*: Total number of features in the dataset
- *num_classes*: Number of classes in classification task. Set to 1 for binary or regression task.
- *cat_idx*: List the indices of the categorical features in your dataset (if there are any).

It is recommended to specify the remaining hyperparameters here as well.

----------------------------

<!-- ![Architecture of the docker container](Docker_architecture.png) -->




## Citation  
If you use this codebase, please cite our work:
```
@article{borisov2021deep,
  title={Deep neural networks and tabular data: A survey},
  author={Borisov, Vadim and Leemann, Tobias and Se{\ss}ler, Kathrin and Haug, Johannes and Pawelczyk, Martin and Kasneci, Gjergji},
  journal={arXiv preprint arXiv:2110.01889},
  year={2021}
}
```
