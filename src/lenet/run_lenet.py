"""LeNet training script.

Usage:
    run_lenet.py [--download]

Options:
    --download  Download the CIFAR-10 dataset before training.
"""

from lenet import LeNet
import torch
import torchvision
from docopt import docopt
from download import download
from helper import (imageshow, train_and_test, predict)
from src.lenet.helper import check_accuracy, check_class_accuracy


def setup_device():
    """ Setup the device used by PyTorch.
    """

    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")

    return device


def main():
    args = docopt(__doc__)

    # Save the device
    device = setup_device()

    if args['--download']:
        download_flag = True
    else:
        download_flag = False

    lenet = LeNet()
    print(lenet)

    trainloader, testloader, classes = download(download_flag)
    # sample images from training set
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    # display images in a grid
    num_images = 4
    imageshow(torchvision.utils.make_grid(images[:num_images]))
    # print labels
    print('    '+'  ||  '.join(classes[labels[j]] for j in range(num_images)))

    model_path = 'cifar_model.pth'
    train_and_test(lenet, trainloader, testloader, model_path)

    lenet_cached = predict(testloader, classes, model_path)

    check_accuracy(lenet_cached, testloader)

    check_class_accuracy(lenet_cached, testloader, classes)



if __name__ == '__main__':
    main()