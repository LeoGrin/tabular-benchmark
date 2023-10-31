from configs.model_configs.skorch_config import skorch_config, skorch_config_default

config_random  = {
    "model__batch_size": {
        "value": "auto"
    },
    "model__es_patience": {
        "value": 16
    },
    "model__lr_scheduler": {
        "values": [False]
    },
    # specified for this benchmark
    "model__module__d_main": {
        "distribution": "q_uniform",
        "min": 16,
        "max": 384
    },
    "model__module__d_multiplier": {
        "value": 2
    },
    "model__lr": {
       #LogUniform[3e-5, 1e-3]
       "distribution": "log_uniform_values",
        "min": 3e-5,
        "max": 1e-3
    },

    "model__optimizer__weight_decay": {
        #{0, LogUniform[1e-6, 1e-3]} #TODO: implement 0
        "distribution": "log_uniform_values",
        "min": 1e-9, #TODO: changed from 1e-6 to mimic 0 choice
        "max": 1e-4 #https://github.com/yandex-research/tabular-dl-tabr/blob/d628ec7e1c0a66011473021034e7dd4a77740112/exp/tabr/why/classif-cat-medium-0-compass/0-tuning/report.json

    },
    #encoder_n_blocks
    "model__module__encoder_n_blocks": {
        "values": [0, 1]
    },
    #predictor_n_blocks
    "model__module__predictor_n_blocks": {
        "values": [1, 2]
    },
    #dropout0
    #I think this correspond to FFN dropout in the paper
    "model__module__dropout0": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.6
    },
    #dropout1
    "model__module__dropout1": {
        #"value": "dropout0", #TODO: not sure about this
        "value": 0.0 #https://github.com/yandex-research/tabular-dl-tabr/blob/d628ec7e1c0a66011473021034e7dd4a77740112/exp/tabr/why/classif-cat-medium-0-compass/0-tuning/report.json
    },
    #context_dropout
    #I think this correspond to attention dropout in the paper
    "model__module__context_dropout": {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.6
    },
    #mixer_normalization
    "model__module__mixer_normalization": {
        "value": "auto",
    },
    #normalization = "LayerNorm"
    #activation = "ReLU"
    "model__module__activation": {
        "value": "ReLU"
    },
    "model__module__normalization": {
        "value": "LayerNorm"
    },
    # num_embeddings
    "module__num_embeddings_temp": {
        "value": "None" #"None" is replaced by None for _temp param #TODO: check
        #model__ is added at the same time when as None replacement
    },
    "use_gpu": {
        "value": True
    }
}
#Defaults for TabR-S
config_default = {
    "model__batch_size": {
        "value": "auto"
    },
    "model__es_patience": {
        "value": 16
    },
    "model__lr_scheduler": {
        "values": [False]
    },
    "model__module__d_main": {
        "value": 256
    },
    "model__module__d_multiplier": {
        "value": 2
    },
    "model__lr": {
        "value": 0.0003121273641315169,
    },
    "model__optimizer__weight_decay": {
        "value": 0.0000012260352006404615,
    },
    #encoder_n_blocks
    "model__module__encoder_n_blocks": {
        "value": 0,
    },
    #predictor_n_blocks
    "model__module__predictor_n_blocks": {
        "value": 1,
    },
    #dropout0
    #I think this correspond to FFN dropout in the paper
    "model__module__dropout0": {
        "value": 0.38852797479169876,
    },
    #dropout1
    "model__module__dropout1": {
        #"value": "dropout0", #TODO: not sure about this
        "value": 0.0 #
    },
    #context_dropout
    #I think this correspond to attention dropout in the paper
    "model__module__context_dropout": {
        "value": 0.38920071545944357,
    },
    #mixer_normalization
    "model__module__mixer_normalization": {
        "value": "auto",
    },
    #normalization = "LayerNorm"
    #activation = "ReLU"
    "model__module__activation": {
        "value": "ReLU"
    },
    "model__module__normalization": {
        "value": "LayerNorm"
    },
    # num_embeddings
    "module__num_embeddings_temp": {
        "value": "None" #"None" is replaced by None for _temp param #TODO: check
        #model__ is added at the same time when as None replacement
    },
    "use_gpu": {
        "value": True
    }
}

config_regression = dict(config_random ,
                                **skorch_config,
                                **{
                                    "model_name": {
                                        "value": "tabr_regressor"
                                    },
                                })

config_regression_default = dict(config_default,
                                **skorch_config_default,
                                **{
                                    "model_name": {
                                        "value": "tabr_regressor"
                                    },
                                })

config_classif = dict(config_random ,
                             **skorch_config,
                             **{
                                 "model_name": {
                                     "value": "tabr"
                                 },
                             })

config_classif_default = dict(config_default,
                             **skorch_config_default,
                             **{
                                 "model_name": {
                                     "value": "tabr"
                                 },
                             })