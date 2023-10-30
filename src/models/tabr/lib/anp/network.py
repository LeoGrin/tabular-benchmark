# fmt: off
# isort: off

from lib.anp.module import *
import torch.nn.functional as F


class LatentModel(nn.Module):
    """
    Latent Model (Attentive Neural Process)
    """

    def __init__(self, d_in, num_hidden, num_self_attention_l, num_cross_attention_l, num_layers_dec):
        super(LatentModel, self).__init__()
        self.latent_encoder = LatentEncoder(num_hidden, num_hidden, input_dim=d_in, num_self_attention_l=num_self_attention_l)
        self.deterministic_encoder = DeterministicEncoder(
            num_hidden, num_hidden, input_dim=d_in, num_self_attention_l=num_self_attention_l, num_cross_attention_l=num_cross_attention_l
        )
        self.decoder = Decoder(num_hidden, input_dim=d_in, num_layers_dec=num_layers_dec)
        # self.BCELoss = nn.BCELoss()
        self.GLoss = nn.GaussianNLLLoss()

    def forward(self, context_x, context_y, target_x, target_y=None):
        num_targets = target_x.size(1)
        
        prior_mu, prior_var, prior = self.latent_encoder(context_x, context_y)
        
        # For training
        if target_y is not None:
            posterior_mu, posterior_var, posterior = self.latent_encoder(target_x, target_y)
            z = posterior
        
        # For Generation
        else:
            z = prior
        
        z = z.unsqueeze(1).repeat(1,num_targets,1) # [B, T_target, H]
        r = self.deterministic_encoder(context_x, context_y, target_x) # [B, T_target, H]
        
        # mu should be the prediction of target y
        mu_pred, sigma_pred = self.decoder(r, z, target_x)
        
        # For Training
        if target_y is not None:
            # get log probability
            # bce_loss = self.BCELoss(t.sigmoid(y_pred), target_y)
            gauss_loss = self.GLoss(target_y, mu_pred, sigma_pred)
            
            
            # get KL divergence between prior and posterior
            kl = self.kl_div(prior_mu, prior_var, posterior_mu, posterior_var)
            
            # maximize prob and minimize KL divergence
            loss = gauss_loss + kl
        
        # For Generation
        else:
            log_p = None
            kl = None
            loss = None
        
        return mu_pred, kl, loss
    
    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (t.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / t.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div
