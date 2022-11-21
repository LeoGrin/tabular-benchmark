import numpy as np

import torch
from torch.utils.data import TensorDataset

import models.deepgbm_lib.config as config

from models.deepgbm_lib.utils.helper import eval_metrics, printMetric

'''

    Train given model (used for Embedding Model and DeepGBM)
    
    Returns:
        - model: trained model
        - optimizer: optimizer used during training

'''

model_path = "deepgbm.pt"


def trainModel(model, train_x, train_y, tree_outputs, test_x, test_y, optimizer,
               train_x_cat=None, test_x_cat=None, epochs=20, early_stopping_rounds=5, save_model=False):
    task = config.config['task']
    device = config.config['device']

    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y)
    tree_outputs = torch.tensor(tree_outputs)

    if train_x_cat is not None:
        train_x_cat = torch.tensor(train_x_cat)
        trainset = TensorDataset(train_x, train_y, tree_outputs, train_x_cat)
    else:
        trainset = TensorDataset(train_x, train_y, tree_outputs)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.config['batch_size'])
    # , shuffle=True, num_workers=2

    min_test_loss = float("inf")
    min_test_loss_idx = 0

    loss_history = []
    val_loss_history = []

    model = model.to(device)

    for epoch in range(epochs):

        running_loss = 0.0
        num_it = 0

        for i, data in enumerate(trainloader, 0):

            # Get data and target from trainloader

            if train_x_cat is not None:
                inputs, target, tree_targets, inputs_cat = data
                inputs_cat = inputs_cat.to(device)
            else:
                inputs, target, tree_targets = data

            inputs, target, tree_targets = inputs.to(device), target.to(device), tree_targets.to(device)

            # Put model in training mode
            model.train()

            # Zero the gradients
            optimizer.zero_grad()

            # forward
            if train_x_cat is not None:
                outputs = model(inputs, inputs_cat)
            else:
                outputs = model(inputs)

            # Compute the loss using the tree outputs
            loss_ratio = max(0.3, config.config['loss_dr'] ** (epoch // config.config['loss_de']))
            loss_val = model.joint_loss(outputs[0], target.float(), outputs[1], tree_targets.float(), loss_ratio)

            # Update gradients
            loss_val.backward()

            # Compute loss for documentation
            loss = model.true_loss(outputs[0], target.float())
            loss_history.append(loss.item())

            # optimize the parameters
            optimizer.step()

            # Update statistics
            running_loss += loss.item()
            num_it += 1

        print("Epoch %d: training loss %.3f" % (epoch + 1, running_loss / num_it))
        running_loss = 0.0

        # Eval Testset
        test_loss, preds = evaluateModel(model, test_x, test_y, test_x_cat)
        metric = eval_metrics(task, test_y, preds)
        printMetric(task, metric, test_loss)

        val_loss_history.append(test_loss)

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            min_test_loss_idx = epoch

        if min_test_loss_idx + early_stopping_rounds < epoch:
            print("Early stopping applies!")
            break

    print('Finished Training')

    if save_model:
        torch.save(model.state_dict(), model_path)

    return model, optimizer, loss_history, val_loss_history


'''

    Evaluate given model (used for Embedding Model and DeepGBM)
    
    Returns:
        - test loss
        - predictions on test data

'''


def evaluateModel(model, test_x, test_y, test_x_cat=None):
    device = config.config['device']

    test_x = torch.tensor(test_x)
    test_y = torch.tensor(test_y)

    if test_x_cat is not None:
        test_x_cat = torch.tensor(test_x_cat)
        testset = TensorDataset(test_x, test_y, test_x_cat)
    else:
        testset = TensorDataset(test_x, test_y)

    testloader = torch.utils.data.DataLoader(testset, batch_size=config.config['test_batch_size'])

    # Put model in evaluation mode
    model = model.to(device)
    model.eval()

    y_preds = []
    sum_loss = 0

    with torch.no_grad():
        for data in testloader:

            # Get data and target from dataloader
            if test_x_cat is not None:
                inputs, target, inputs_cat = data
                inputs_cat = inputs_cat.to(device)
            else:
                inputs, target = data

            inputs, target = inputs.to(device), target.to(device)

            # Calculate outputs 
            if test_x_cat is not None:
                outputs = model(inputs, inputs_cat)[0]
            else:
                outputs = model(inputs)[0]

            y_preds.append(outputs.cpu().detach().numpy())

            # Compute loss
            loss = model.true_loss(outputs, target.float()).item()
            sum_loss += loss * target.shape[0]

    return sum_loss / test_x.shape[0], np.concatenate(y_preds)


'''
    Make predictions on new data
    
    Returns:
        - predictions for input data

'''


def makePredictions(model, test_x, test_cat):
    device = config.config['device']

    testset = TensorDataset(torch.tensor(test_x), torch.tensor(test_cat))
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.config['test_batch_size'])

    # Put model in evaluation mode
    model = model.to(device)
    model.eval()

    y_preds = []

    with torch.no_grad():
        for data in testloader:
            inputs, inputs_cat = data
            inputs, inputs_cat = inputs.to(device), inputs_cat.to(device)

            outputs = model(inputs, inputs_cat)[0]
            y_preds.append(outputs.cpu().detach().numpy())

    return np.concatenate(y_preds, axis=0)
