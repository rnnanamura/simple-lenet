import torch
import torchvision
import torchvision.transforms as transforms

from src import lenet

torch.use_deterministic_algorithms(True)

def download(download):
    # The mean and std are kept as 0.5 for normalizing
    # pixel values as the pixel values are originally
    # in the range 0 to 1
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, 4),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True, download=download, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=8, shuffle=True)
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5),
                                                              (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=10000, shuffle=False)
    # ordering is important
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
               'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes
