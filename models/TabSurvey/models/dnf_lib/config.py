import os
import copy

from sklearn.metrics import mean_squared_error

from models.dnf_lib.Utils.file_utils import create_dir


my_config = {
    'n_formulas': 64,
    'orthogonal_lambda': 0.,
    'elastic_net_beta': 0.1,
    'random_seed': 1306,
    
    'input_dim': 54,
    'output_dim': 7,
    'translate_label_to_one_hot': True,

    # 'XGB_objective': "multi:softprob", #"binary:logistic",
    
    'experiment_number': 1,

    'GPU': '0'
}

# Values for hyperparameter search
DNFNet_grid_params = {
    'n_formulas': [3072, 2048, 1024, 512, 256, 128, 64],
    'orthogonal_lambda': [0.],
    'elastic_net_beta': [1.6, 1.3, 1., 0.7, 0.4, 0.1],
}


shared_config = {
    'n_conjunctions_arr': [6, 9, 12, 15],
    'conjunctions_depth_arr': [2, 4, 6],
    'keep_feature_prob_arr': [0.1, 0.3, 0.5, 0.7, 0.9],

    'initial_lr': 5e-2,
    'lr_decay_factor': 0.5,
    'lr_patience': 10,
    'min_lr': 1e-6,

    'early_stopping_patience': 20,
    'epochs': 1000,
    'batch_size': 128, #2048,

    'apply_standardization': True,

    'save_weights': True,
    'starting_epoch_to_save': 0,
    'models_module_name': 'models.dnf_lib.DNFNet.DNFNetModels',
    'models_dir': 'models/dnf_lib/DNFNet/DNFNetModels',
    
    'model_number': 1  # That is the whole DNF with localization and feature selection
}

score_config = {
    'score_metric': mean_squared_error,
    'score_increases': False,
    # 'XGB_eval_metric': 'acc',
}


def merge_configs(config_dst, config_src):
    config_dst = copy.deepcopy(config_dst)
    config_src = copy.deepcopy(config_src)
    for key, value in config_src.items():
        config_dst[key] = value
    return config_dst


def get_config(competition_name, model):
    
    shared_config['experiments_dir'] = 'output/DNFNet/experiments/{}'.format(competition_name)
    output_dir = os.path.join(shared_config['experiments_dir'], 'grid_search')
    create_dir(shared_config['experiments_dir'])
    shared_config['competition_name'] = competition_name
    shared_config['model_name'] = model
    
    config = merge_configs(shared_config, my_config)
    create_dir(output_dir)
        
    return config, score_config
