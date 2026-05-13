"""AlexNet training script.

Usage:
    run_alexnet.py [--imagedir IMAGEDIR] [--model MODEL]

Options:
    --imagedir IMAGEDIR  image dataset directory.
    --model MODEL  model name either AlexNet or VGG13
"""
import ast

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import models
from docopt import docopt
from datasets import load_data_alexnet, load_data_vgg13
from helper import (imageshow, finetune_model, visualize_predictions_alexnet, visualize_predictions_vgg13)

#torch.use_deterministic_algorithms(True)

def main():
    args = docopt(__doc__)
    ddir = args['--imagedir']


    model_name = args['--model']
    if model_name == 'AlexNet':
        dloaders, dvc, dset_sizes, classes = load_data_alexnet(ddir)
        # Generate one train dataset batch
        imgs, cls = next(iter(dloaders['train']))
        # Generate a grid from batch
        grid = torchvision.utils.make_grid(imgs)
        imageshow(grid, text=[classes[c] for c in cls])

        model_finetune = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        print(model_finetune.features)

        print(model_finetune.classifier)
        # change the last layer from 1000 classes to 2 classes
        model_finetune.classifier[6] = nn.Linear(4096, len(classes))

        loss_func = nn.CrossEntropyLoss()
        optim_finetune = optim.SGD(model_finetune.parameters(), lr=0.0001)

        # train (fine-tune) and validate the model
        model_finetune = finetune_model(model_finetune, loss_func, optim_finetune, dloaders, dvc, dset_sizes, epochs=10)
        visualize_predictions_alexnet(model_finetune, dloaders, dvc, classes, max_num_imgs=10)
    elif model_name == 'VGG13':
        torch.use_deterministic_algorithms(True)
        dloaders, dvc, dset_sizes = load_data_vgg13(ddir)
        with open('./imagenet1000_clsidx_to_labels.txt') as f:
            classes_data = f.read()
        classes_dict = ast.literal_eval(classes_data)
        print({k: classes_dict[k] for k in list(classes_dict)[:5]})

        model_finetune = models.vgg13(pretrained=True)
        print(model_finetune.features)
        print(model_finetune.classifier)
        visualize_predictions_vgg13(model_finetune, dloaders, dvc, classes_dict, max_num_imgs=10)

if __name__ == '__main__':
    main()