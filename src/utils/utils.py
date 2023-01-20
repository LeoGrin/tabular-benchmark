import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import itertools
import numbers



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



def numpy_to_dataset(X_train, Y_train, X_test, Y_test, batch_size, X_val=None, Y_val = None, device=None):
    tensor_x_train, tensor_y_train = torch.tensor(X_train, device=device, dtype=torch.float), torch.tensor(Y_train, device=device, dtype=torch.long)

    tensor_x_test, tensor_y_test = torch.tensor(X_test, device=device, dtype=torch.float), torch.tensor(Y_test, device=device, dtype=torch.long)

    train_dataset = TensorDataset(tensor_x_train, tensor_y_train)  # create your datset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # create your dataloader

    test_dataset = TensorDataset(tensor_x_test, tensor_y_test)  # create your datset
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)  # create your dataloader
    if (not X_val is None) and (not Y_val is None):
        tensor_x_val, tensor_y_val = torch.tensor(X_val, device=device, dtype=torch.float), torch.tensor(Y_val, device=device, dtype=torch.long)
        val_dataset = TensorDataset(tensor_x_val, tensor_y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        return train_dataloader, test_dataloader, val_dataloader

    return train_dataloader, test_dataloader

def remove_key_from_dict(dict, key_to_del):
    return {key: val for key, val in dict.items() if key != key_to_del}

def remove_keys_from_dict(dict, keys_to_del):
    return {key: val for key, val in dict.items() if key not in keys_to_del}

def merge_dics(dics):
    #concatenate values for each keys
    keys = []
    for dic in dics:
        keys.extend(list(dic.keys()))
    keys = np.unique(keys)
    merged_dics = {}
    for key in keys:
        values = []
        for dic in dics: #get type
            if key in dic.keys():
                value_example = dic[key]
        for dic in dics:
            if key in dic.keys():
                values.append(dic[key])
            else:
                if isinstance(value_example, numbers.Number): #replace by a placeholder depending on type
                    values.append(np.nan)
                else:
                    values.append("none")
        values = np.unique(values)
        if len(values) == 1:
            merged_dics[key] = values[0]
        elif len(values) == 0:
            raise ValueError
        elif len(values) > 1:
            if type(values[0]) == list:
                merged_dics[key] = list(itertools.chain(*values))
            else:
                merged_dics[key] = values
    return merged_dics

