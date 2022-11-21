config = {}

# Common Parameters
config['task'] = 'binary'  # 'regression', 'classification'
config['batch_size'] = 128
config['test_batch_size'] = 128
config['l2_reg'] = 1e-6
config['l2_reg_opt'] = 5e-4
config['lr'] = 1e-3
config['epochs'] = 30
config['early_stopping'] = 3
config['loss_de'] = 2
config['loss_dr'] = 0.7

config['device'] = 'cpu'

# Preprocessing parameters
config['bins'] = 32
config['rate'] = 0.9  # train test split rate
config['threshold'] = 10

# GBDT parameters
config['maxleaf'] = 64
config['num_slices'] = 5
config['n_feature'] = 128
config['n_clusters'] = 10
config['tree_lr'] = 0.1
config['n_trees'] = 100

# Embedding parameters
config['embsize'] = 20
config['emb_lr'] = 1e-3
config['emb_epochs'] = 3

# GDBT2NN Parameters
config['tree_layers'] = [100, 100, 100, 50] + [config['embsize']]

# CatNN Parameters
config['embedding_size'] = 20  # 4
config['cate_layers'] = [16, 16]

# Online parameters
config['online_epochs'] = 1
config['online_bz'] = 4096
config['num_splits'] = 5
