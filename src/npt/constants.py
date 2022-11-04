# ########## Classification/Regression Mask Modes  ##########
# #### Stochastic Label Masking ####
# On these labels, stochastic masking may take place at training time, vali-
# dation time, or at test time.
DATA_MODE_TO_LABEL_BERT_MODE = {
    'train': ['train'],
    'val': ['train'],
    'test': ['train', 'val'],
}

# However, even when we do stochastic label masking, some labels will be
# masked out deterministically, to avoid information leaks.
DATA_MODE_TO_LABEL_BERT_FIXED = {
    'train': ['val', 'test'],
    'val': ['val', 'test'],
    'test': ['test'],
}
