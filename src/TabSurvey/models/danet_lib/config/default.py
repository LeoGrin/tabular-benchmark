from yacs.config import CfgNode as Node
cfg = Node()
cfg.seed = 324
cfg.dataset = 'forest_cover_type'
cfg.task = 'classification'
cfg.resume_dir = ''
cfg.logname = ''

cfg.model = Node()
cfg.model.base_outdim = 64
cfg.model.k = 5
cfg.model.drop_rate = 0.1
cfg.model.layer = 20

cfg.fit = Node()
cfg.fit.lr = 0.008
cfg.fit.max_epochs = 2
cfg.fit.patience = 20
cfg.fit.batch_size = 128
cfg.fit.virtual_batch_size = 256
