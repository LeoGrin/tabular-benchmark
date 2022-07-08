import importlib
import os
import copy
from shutil import copyfile
from models.dnf_lib.Utils.file_utils import create_dir, write_dict_to_file, read_dict_from_file, delete_file


def write_config(config, run_dir):
    d = copy.deepcopy(config)
    if 'loc_dict' in d:
        del d['loc_dict']
    if 'mask' in d:
        d['mask'] = list(d['mask'])
    write_dict_to_file(d, run_dir)


def create_experiment_directory(config, copy_model=True, return_sub_dirs=False):
    experiment_dir = config['experiments_dir'] + '/{}'.format(config['experiment_number'])
    weights_dir = os.path.join(experiment_dir, "weights")
    logs_dir = os.path.join(experiment_dir, "logs")
    create_dir(experiment_dir)
    create_dir(weights_dir)
    create_dir(logs_dir)
    write_config(config, experiment_dir + '/configuration.json')
    if copy_model:
        copyfile(config['models_dir'] + '/model{}.py'.format(config['model_number']), experiment_dir + '/model.py')

    if return_sub_dirs:
        return experiment_dir, weights_dir, logs_dir
    else:
        return experiment_dir


def create_model(config, models_module_name):
    full_class_name = "{}.model{}".format(models_module_name, config['model_number'])
    module = importlib.import_module(full_class_name)
    model_class = getattr(module, "MNN")
    return model_class(config)


def read_config(config_path):
    config = read_dict_from_file(config_path)
    return config


def load_config_from_experiment_dir(config):
    experiment_dir = config['experiments_dir'] + '/{}'.format(config['experiment_number'])
    loaded_config = read_config(experiment_dir + '/configuration.json')
    return loaded_config, experiment_dir


def load_config(experiment_number, experiments_dir):
    experiment_dir = experiments_dir + '/{}'.format(experiment_number)
    loaded_config = read_config(experiment_dir + '/configuration.json')
    return loaded_config


def create_all_FCN_layers_with_fixed_width(depth_arr, width_arr):
    layers = []
    for d in depth_arr:
        for w in width_arr:
            layers.append([w] * d)
    return layers


def create_all_FCN_layers_with_factor_2_reduction(depth_arr, width_arr):
    layers = []
    for initial_w in width_arr:
        for d in depth_arr:
            net = []
            for i in range(d):
                net.append(int(initial_w / (2**i)))
            layers.append(net)
    return layers


def create_all_FCN_layers_grid(depth_arr, width_arr):
    depth_arr_copy = copy.deepcopy(depth_arr)
    if 1 in depth_arr_copy:
        depth_arr_copy.remove(1)
    return create_all_FCN_layers_with_fixed_width(depth_arr_copy, width_arr) + create_all_FCN_layers_with_factor_2_reduction(depth_arr, width_arr)
