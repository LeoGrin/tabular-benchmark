import sys
sys.path.append(".")
from configs.all_model_configs import model_keyword_dic
import numpy as np
from tab_models.sklearn.default_params import DefaultParams
from tab_models import utils

# for david's model
def convert_raw_mlp_params(params, is_classification):
    special_param_names = ['num_emb_type', 'use_front_scale']
    not_special_params = {key: value for key, value in params.items() if key not in special_param_names}
    default_params = DefaultParams.MLP_TD_CLASS if is_classification else DefaultParams.MLP_TD_REG
    config = utils.update_dict(default_params, not_special_params)
    num_emb_type = params.get('num_emb_type', None)
    if num_emb_type == 'none':
        config['use_plr_embeddings'] = False
    elif num_emb_type == 'pl-densenet':
        pass  # this is already the default
    elif num_emb_type == 'plr':
        config['plr_act_name'] = 'relu'
        config['plr_use_densenet'] = False
    elif num_emb_type is None:
        pass  # also use the default
    else:
        raise ValueError(f'Unknown num_emb_type "{num_emb_type}"')

    if not config.get('use_front_scale', True):
        config['first_layer_config'] = dict(block_str='w-b-a-d')

    return config

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def create_model(config, categorical_indicator, cat_cardinalities=None, num_features=None, id=None, cat_dims=None):
    print(model_keyword_dic)
    model_function = model_keyword_dic[config["model_name"]]
    model_config = {}
    for key in config.keys():
        if key.startswith("model__"):
            model_config[key[len("model__"):]] = config[key]
    if config["model_type"] == "skorch":
        model_config["categorical_indicator"] = categorical_indicator
        model_config["categories"] = cat_cardinalities
        return model_function(**model_config, id=id)
    elif config["model_type"] == "sklearn":
        if config["model_name"].startswith("hgbt"):
            # Use native support for categorical variables
            model_config["categorical_features"] = categorical_indicator
        return model_function(**model_config)
    elif config["model_type"] == "david":
        model_config = convert_raw_mlp_params(model_config, is_classification = not "regressor" in config["model_name"])
        return model_function(**model_config)
    elif config["model_type"] == "tab_survey":
        args_dic = {}
        params_dic = {}
        for key in model_config.keys():
            if key.startswith("args__"):
                args_dic[key[len("args__"):]] = model_config[key]
            elif key.startswith("params__"):
                params_dic[key[len("params__"):]] = model_config[key]
        args_dic["model_id"] = id
        args_dic["cat_idx"] = np.where(categorical_indicator)[0]
        args_dic["cat_dims"] = cat_dims
        args_dic["num_features"] = num_features
        args_dic["dataset"] = config["data__keyword"] #TODO
        print(args_dic)
        print(params_dic)
        args = AttrDict()
        params = AttrDict()
        args.update(args_dic)
        params.update(params_dic)
        return model_function(args=args, params=params)
