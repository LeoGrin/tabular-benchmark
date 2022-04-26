from utils.keyword_to_function_conversion import convert_keyword_to_function


def create_model(config, id=None):
    model_function = convert_keyword_to_function(config["model_name"])
    model_config = {}
    for key in config.keys():
        if key.startswith("model__"):
            model_config[key[len("model__"):]] = config[key]
    if config["model_type"] == "skorch":
        return model_function(**model_config, id=id)
    elif config["model_type"] == "sklearn":
        return model_function(**model_config)
