"""Rand Wire RNN

Usage:
    run_rand_wire_rnn.py [--train] [--test]
    run_rand_wire_rnn.py (-h | --help)
    run_rand_wire_rnn.py (-v | --version)

Options:
    --train  train the model.
    --test  test the model.
"""

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
from docopt import docopt
from helper import load_dataset, train, accuracy, plot_results, num_model_params
from models import RandWireNNModel


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.use_deterministic_algorithms(True)

def main():
    args = docopt(__doc__)
    # %%
    num_epochs = 5
    graph_probability = 0.7
    node_channel_count = 64
    num_nodes = 16
    lrate = 0.1
    batch_size = 64
    train_mode = True


    train_dataloader, test_dataloader = load_dataset(batch_size)

    if args["--train"]:
        # %%
        rand_wire_model = RandWireNNModel(num_nodes, graph_probability, node_channel_count, node_channel_count,
                                          train_mode).to(device)

        optim_module = optim.SGD(rand_wire_model.parameters(), lr=lrate, weight_decay=1e-4, momentum=0.8)
        loss_func = nn.CrossEntropyLoss().to(device)

        epochs = []
        test_accuracies = []
        training_accuracies = []
        training_losses = []
        best_test_accuracy = 0

        start_time = time.time()
        for ep in range(1, num_epochs + 1):
            epochs.append(ep)
            training_loss, training_accuracy = train(rand_wire_model, train_dataloader, optim_module, loss_func, ep,
                                                     lrate, batch_size)
            test_accuracy = accuracy(rand_wire_model, test_dataloader)
            test_accuracies.append(test_accuracy)
            training_losses.append(training_loss)
            training_accuracies.append(training_accuracy)
            print('test acc: {0:.2f}%, best test acc: {1:.2f}%'.format(test_accuracy, best_test_accuracy))

            if best_test_accuracy < test_accuracy:
                model_state = {
                    'model': rand_wire_model.state_dict(),
                    'accuracy': test_accuracy,
                    'ep': ep,
                }
                if not os.path.isdir('model_checkpoint'):
                    os.mkdir('model_checkpoint')
                model_filename = "ch_count_" + str(node_channel_count) + "_prob_" + str(graph_probability)
                torch.save(model_state, './model_checkpoint/' + model_filename + 'ckpt.t7')
                best_test_accuracy = test_accuracy
                plot_results(epochs, training_losses, training_accuracies, test_accuracies)
            print("model train time: ", time.time() - start_time)
            print("total model params: ", num_model_params(rand_wire_model))
    if args["--test"]:
        # %%
        if os.path.exists("./model_checkpoint"):
            rand_wire_nn_model = RandWireNNModel(num_nodes, graph_probability, node_channel_count, node_channel_count,
                                                 train_mode=False).to(device)
            model_filename = "ch_count_" + str(node_channel_count) + "_prob_" + str(graph_probability)
            model_checkpoint = torch.load('./model_checkpoint/' + model_filename + 'ckpt.t7')
            rand_wire_nn_model.load_state_dict(model_checkpoint['model'])
            last_ep = model_checkpoint['ep']
            best_model_accuracy = model_checkpoint['accuracy']
            print(f"best model accuracy: {best_model_accuracy}%, last epoch: {last_ep}")

            rand_wire_nn_model.eval()
            success = 0
            for test_data, test_label in test_dataloader:
                test_data, test_label = test_data.to(device), test_label.to(device)
                pred_raw = rand_wire_nn_model(test_data)
                pred = pred_raw.data.max(1)[1]
                success += pred.eq(test_label.data).sum()
            print(f"test accuracy: {float(success) * 100. / len(test_dataloader.dataset)} %")

        else:
            assert False, "File not found. Please check again."

if __name__ == "__main__":
    main()