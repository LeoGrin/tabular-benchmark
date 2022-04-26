import torch
from utils.skorch_utils import create_mlp_skorch_regressor

def start_from_pretrained_model(model_id, wandb_run, pretrained_model_filename, noise_std, **kwargs):
    model = create_mlp_skorch_regressor(model_id, wandb_run,
                           **kwargs)  # create a different checkpointing file for each run to avoid conflict (I'm not 100% sure it's necessary)
    # model = pickle.load(open(
    #    'saved_models/regression_synthetic_{}_{}_{}_{}_mlp_pickle.pkl'.format(5000, 0, 16,
    #                                                                   iter), 'rb'))
    model.initialize()
    model.load_params(
        f_params=pretrained_model_filename + ".params")
    model.module_.no_reinitialize = True


    # Add noise
    state_dict = model.module_.state_dict()
    for param in state_dict.keys():
        print("#########")
        if param.startswith("fc_layers") or param.startswith("input_layer") or param.startswith("output_layer"):
            state_dict[param] = model.module_.state_dict()[param] + noise_std * torch.randn(model.module_.state_dict()[param].shape)
            model.module_.load_state_dict(state_dict)

    return model





