import numpy as np

import torch
import torch.optim as optim

from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error

import models.deepgbm_lib.config as config


def eval_metrics(task, true, pred):
    if task == 'binary':
        logloss = log_loss(true.astype(np.float64), pred.astype(np.float64))
        auc = roc_auc_score(true, pred)
        # error = 1-sklearn.metrics.accuracy_score(true,(pred+0.5).astype(np.int32))
        return (logloss, auc)#, error)
    elif task == 'regression':
        mseloss = mean_squared_error(true, pred)
        return mseloss
    else:
        print ("Classification not yet implemented")
        # TODO: Implement classification
        
        
def printMetric(task, metric, test_loss):
    if task == 'binary':
        test_error, test_auc = metric
        print ("Test Loss of %f, Test AUC of %f" % (test_loss, test_auc))
    elif task == 'regression':
        print ("Test Loss of %f, Test MSE of %f" % (test_loss, metric))
    else:
        print ("Classification not yet implemented")
        # TODO: implement classification


class AdamW(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, model_decay_opt=None,
                 weight_decay_opt=None, key_opt=''):
        super(AdamW, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=0, amsgrad=amsgrad)
        self.weight_decay = weight_decay
        self.weight_decay_opt = weight_decay_opt
        self.model_decay_opt = model_decay_opt
        self.key_opt = key_opt
    def step(self, closure=None):
        loss = super(AdamW, self).step(closure)
        for group in self.param_groups:
            if self.weight_decay != 0:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.data.add_(-self.weight_decay, p.data)
        if self.model_decay_opt is not None:
            for name, parameter in self.model_decay_opt.named_parameters():
                if self.key_opt in name:
                    if parameter.grad is not None:
                        parameter.data.add_(-self.weight_decay_opt, parameter.data)
        return loss
    
    
def outputFromEmbeddingModel(emb_model, leaf_preds, label_size, n_models):
    trainloader = torch.utils.data.DataLoader(leaf_preds, batch_size=config.config['batch_size'])
    #, shuffle=True, num_workers=2

    emb_model.eval()
    y_preds = []

    with torch.no_grad():
        for data in trainloader:

            # calculate outputs by running images through the network
            outputs = emb_model.lastlayer(data).data.cpu().numpy()
            y_preds.append(outputs)

        y_preds = np.concatenate(y_preds, 0)


    output_w = emb_model.bout.weight.data.cpu().numpy().reshape(n_models*config.config['embsize'], label_size)
    output_b = np.array(emb_model.bout.bias.data.cpu().numpy().sum())
    
    return output_w, output_b, y_preds
