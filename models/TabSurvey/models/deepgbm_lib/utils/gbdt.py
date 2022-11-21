'''

    Train a GBDT and distill the knowlegde from it

'''

import numpy as np

import lightgbm as lgb

from models.deepgbm_lib.utils.tree_model_interpreter import ModelInterpreter

import models.deepgbm_lib.config as config



def TrainGBDT(train_x, train_y, test_x, test_y):
    
    '''
    Parameters: trainings and test data, hyperparmeters for the GBDT
    Returns: the trained tree
    '''
    
    task = config.config['task']
    
    num_class = 1
    
    if task == 'regression':
        objective = "regression"
        metric = "mse"
        boost_from_average = True
    elif task == 'binary':
        objective = "binary"
        metric = "auc"
        boost_from_average = True
    else:
        print ("Classification not yet implemented!")
        # TODO: implement classification
        
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'num_class': num_class,
        'objective': objective,
        'metric': metric,
        'num_leaves': config.config['maxleaf'],
        'min_data': 40,
        'boost_from_average': boost_from_average,
        'num_threads': 6,
        'feature_fraction': 0.8,
        'bagging_freq': 3,
        'bagging_fraction': 0.9,
        'learning_rate': config.config['tree_lr'],
    }
    
    # define the datasets
    lgb_train = lgb.Dataset(train_x, train_y.reshape(-1), params=params)
    lgb_eval = lgb.Dataset(test_x, test_y.reshape(-1), reference=lgb_train)
    
    # train light GB
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=config.config['n_trees'],
                    valid_sets=lgb_eval,
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(False)])
    
    # Make predictions on train data
    #preds = gbm.predict(train_x, raw_score=True)
    #preds = preds.astype(np.float32)
    return gbm #, preds




def SubGBDTLeaf_cls(train_x, test_x, gbm):
    
    '''
    
        Distill the knowlegde from the trees
        Return: used features, new y train data, leaf predicitions, test leaf predictions, tree mean
    
    '''
    
    num_slices = config.config['num_slices']
    
    # Get dims
    max_n_features = train_x.shape[1]
    n_samples_train = train_x.shape[0]
    
    # Predict train and test data
    leaf_preds = gbm.predict(train_x, pred_leaf=True)
    test_leaf_preds = gbm.predict(test_x, pred_leaf=True)
    
    # Get number of trees used
    n_trees = leaf_preds.shape[1]

    step = max(int((n_trees + num_slices - 1) // num_slices), 1)
    leaf_output = np.zeros([n_trees, config.config['maxleaf']], dtype=np.float32)
    
    # Get leaf outputs from all trees
    for tid in range(n_trees):
        num_leaf = np.max(leaf_preds[:,tid]) + 1
        for lid in range(num_leaf):
            leaf_output[tid][lid] = gbm.get_leaf_output(tid, lid)
            
    
    # Interprete the trees
    modelI = ModelInterpreter(gbm)
    clusterIdx = modelI.EqualGroup(num_slices)
    treeI = modelI.trees
    
    for n_idx in range(num_slices):
        
        # Get current tree indices
        tree_indices = np.where(clusterIdx == n_idx)[0]
        
        #trees = {}
        #tid = 0
        #for jdx in tree_indices:
        #    trees[str(tid)] = treeI[jdx].raw
        #    tid += 1
        #tree_num = len(tree_indices)
        #layer_num = 1
        #xi = []
        #xi_fea = set()
        
        # Count the gain from all features from the trees
        all_hav = {} 
        for tree in tree_indices:
            for kdx, f in enumerate(treeI[tree].feature):
                if f == -2:
                    continue
                if f not in all_hav:
                    all_hav[f] = 0
                all_hav[f] += treeI[tree].gain[kdx]
                
        # Convert to sorted list of tupels (feature, gain)        
        all_hav = sorted(all_hav.items(), key=lambda kv: -kv[1])
        
        # Take the ordered list of features, only the n_feature-most important ones
        used_features = [item[0] for item in all_hav[:config.config['n_feature']]]
        
        #used_features_set = set(used_features)
        # This is important! It fills the used_features with this dummy value for all entrys having the same size
        for kdx in range(max(0, config.config['n_feature'] - len(used_features))):
            used_features.append(max_n_features)
            
        # Get leaf predicition from current trees
        cur_leaf_preds = leaf_preds[:, tree_indices]
        cur_test_leaf_preds = test_leaf_preds[:, tree_indices]
        
        # Compute new train y values
        new_train_y = np.zeros(train_x.shape[0])
        for jdx in tree_indices:
            new_train_y += np.take(leaf_output[jdx,:].reshape(-1), leaf_preds[:,jdx].reshape(-1))
        new_train_y = new_train_y.reshape(-1,1).astype(np.float32)
        
        # Compute the means of the leaf outputs
        tree_mean = np.mean(np.take(leaf_output, tree_indices,0))
        #total_mean = np.mean(leaf_output)
               
        yield used_features, new_train_y, cur_leaf_preds, cur_test_leaf_preds, tree_mean #, total_mean

        
        
        
        
def get_infos_from_gbms(gbms, min_len_features):
    
    '''
    
        Save and adapt all knowledge from gbms
        
        Return: used_features, tree_outputs, leaf_preds, test_leaf_preds, group_average, max_ntree_per_split
    
    '''

    used_features = []
    tree_outputs = []
    leaf_preds = []
    test_leaf_preds = []
    group_average = []

    #min_len_features = train_x.shape[1]
    max_ntree_per_split = 0

    # Save gbms output in lists
    for used_feature, new_train_y, leaf_pred, test_leaf_pred, avg in gbms:
        used_features.append(used_feature)
        min_len_features = min(min_len_features, len(used_feature))
        tree_outputs.append(new_train_y)
        leaf_preds.append(leaf_pred)
        test_leaf_preds.append(test_leaf_pred)
        group_average.append(avg)
        max_ntree_per_split = max(max_ntree_per_split, leaf_pred.shape[1])
        
        
    n_models = len(used_features)

    for i in range(n_models):
        used_features[i] = sorted(used_features[i][:min_len_features])

    group_average = np.asarray(group_average).reshape(n_models, 1, 1)


    # Increase leaf preds array to max_ntree_per_split size for all models

    for i in range(n_models):
        if leaf_preds[i].shape[1] < max_ntree_per_split:
            leaf_preds[i] = np.concatenate([leaf_preds[i] + 1, 
                                np.zeros([leaf_preds[i].shape[0],
                                max_ntree_per_split-leaf_preds[i].shape[1]],
                                dtype=np.int32)], axis=1)
            test_leaf_preds[i] = np.concatenate([test_leaf_preds[i] + 1, 
                                np.zeros([test_leaf_preds[i].shape[0],
                                max_ntree_per_split-test_leaf_preds[i].shape[1]],
                                dtype=np.int32)], axis=1)

    leaf_preds = np.concatenate(leaf_preds, axis=1)
    test_leaf_preds = np.concatenate(test_leaf_preds, axis=1)
    
    return used_features, tree_outputs, leaf_preds, test_leaf_preds, group_average, max_ntree_per_split