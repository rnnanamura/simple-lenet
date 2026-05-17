#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchviz import make_dot

import os
import time
import yaml
import random
import networkx as nx
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.use_deterministic_algorithms(True)

#%%
def plot_results(list_of_epochs, list_of_train_losses, list_of_train_accuracies, list_of_val_accuracies):
    plt.figure(figsize=(20, 9))
    plt.subplot(1, 2, 1)

    plt.plot(list_of_epochs, list_of_train_losses, label='training loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(list_of_epochs, list_of_train_accuracies, label='training accuracy')
    plt.plot(list_of_epochs, list_of_val_accuracies, label='validation accuracy')
    plt.legend()
    if not os.path.isdir('./result_plots'):
        os.makedirs('./result_plots')
    plt.savefig('./result_plots/accuracy_plot_per_epoch.jpg')
    plt.close()

#%%
def set_lr(optim, epoch_num, lrate):
    """adjusts lr to starting lr thereafter reduced by 10% at every 20 epochs"""
    lrate = lrate * (0.1 ** (epoch_num // 20))
    for params in optim.param_groups:
        params['lr'] = lrate

#%%
def train(model, train_dataloader, optim, loss_func, epoch_num, lrate, batch_size):
    model.train()
    loop_iter = 0
    training_loss = 0
    training_accuracy = 0
    for training_data, training_label in train_dataloader:
        set_lr(optim, epoch_num, lrate)
        training_data, training_label = training_data.to(device), training_label.to(device)
        optim.zero_grad()
        pred_raw = model(training_data)
        curr_loss = loss_func(pred_raw, training_label)
        curr_loss.backward()
        optim.step()
        training_loss += curr_loss.item()
        pred = pred_raw.data.max(1)[1]

        curr_accuracy = float(pred.eq(training_label.data).sum()) * 100. / len(training_data)
        training_accuracy += curr_accuracy
        loop_iter += 1
        if loop_iter % 100 == 0:
            print(f"epoch {epoch_num}, loss: {curr_loss.data}, accuracy: {curr_accuracy}")

    data_size = len(train_dataloader.dataset) // batch_size
    return training_loss / data_size, training_accuracy / data_size

#%%
def accuracy(model, test_data_loader):
    model.eval()
    success = 0
    with torch.no_grad():
        for test_data, test_label in test_data_loader:
            test_data, test_label = test_data.to(device), test_label.to(device)
            pred_raw = model(test_data)
            pred = pred_raw.data.max(1)[1]
            success += pred.eq(test_label.data).sum()

    return float(success) * 100. / len(test_data_loader.dataset)

#%%
def load_dataset(batch_size):
    transform_train_dataset = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4983, 0.4795, 0.4382), (0.2712, 0.2602, 0.2801)),
    ])

    transform_test_dataset = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4983, 0.4795, 0.4382), (0.2712, 0.2602, 0.2801)),
    ])
    train_dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10('dataset', transform=transform_train_dataset, train=True, download=True),
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10('dataset', transform=transform_test_dataset, train=False),
        batch_size=batch_size,
        shuffle=False
    )
    return train_dataloader, test_dataloader


    # %%


def initialize_weights(layer):
    if isinstance(layer, nn.Conv2d):
        torch.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

#%%
def num_model_params(model_obj):
    num_params = 0
    for l in list(model_obj.parameters()):
        l_p = 1
        for p in list(l.size()):
            l_p *= p
        num_params += l_p
    return num_params
