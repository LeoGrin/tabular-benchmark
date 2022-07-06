from utils.keyword_to_function_conversion import convert_keyword_to_function
import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def create_model(config, categorical_indicator, num_features=None, id=None, cat_dims=None):
    model_function = convert_keyword_to_function(config["model_name"])
    model_config = {}
    for key in config.keys():
        if key.startswith("model__"):
            model_config[key[len("model__"):]] = config[key]
    if config["model_type"] == "skorch":
        model_config["categorical_indicator"] = categorical_indicator
        return model_function(**model_config, id=id)
    elif config["model_type"] == "sklearn":
        if config["model_name"].startswith("hgbt"):
            model_config["categorical_features"] = categorical_indicator
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
