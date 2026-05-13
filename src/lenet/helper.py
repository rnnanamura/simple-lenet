import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from lenet import LeNet

def train(net, trainloader, optim, epoch):
    # initialize loss
    loss_total = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # ip refers to the input images, and ground_truth
        # refers to the output classes the images belong to
        ip, ground_truth = data
        # zero the parameter gradients
        optim.zero_grad()
        # forward-pass + backward-pass + optimization -step
        op = net(ip)
        loss = nn.CrossEntropyLoss()(op, ground_truth)
        loss.backward()
        optim.step()
        # update loss
        loss_total += loss.item()
        # print loss statistics
        if ( i +1) % 1000 == 0:
            # print at the interval of 1000 mini-batches
            print('[Epoch number : %d, Mini-batches: %5d] \
                  loss: %.3f' % (epoch + 1, i + 1,
                                 loss_total / 200))
            loss_total = 0.0

def test(net, testloader):
    success = 0
    counter = 0
    with torch.no_grad():
        for data in testloader:
            im, ground_truth = data
            op = net(im)
            _, pred = torch.max(op.data, 1)
            counter += ground_truth.size(0)
            success += (pred == ground_truth).sum().item()
    print('LeNet accuracy on 10000 images from test dataset: %d %%'\
        % (100 * success / counter))

def train_and_test(lenet, trainloader, testloader, model_path):
    # define optimizer
    optim = torch.optim.Adam(lenet.parameters(), lr=0.001)
    # training loop over the dataset multiple times
    for epoch in range(50):
        train(lenet, trainloader, optim, epoch)
        print()
        test(lenet, testloader)
        print()
    print('Finished Training')

    if model_path is not None:
        torch.save(lenet.state_dict(), model_path)


# define a function that displays an image
def imageshow(image):
    # un-normalize the image
    image = image/2 + 0.5
    npimage = image.numpy()
    plt.imshow(np.transpose(npimage, (1, 2, 0)))
    plt.show()

def predict(testloader, classes, model_path):
    # load test dataset images
    d_iter = iter(testloader)
    im, ground_truth = next(d_iter)
    # print images and ground truth
    imageshow(torchvision.utils.make_grid(im[:4]))
    print('Label:      ', ' '.join('%5s' %
                                   classes[ground_truth[j]]
                                   for j in range(4)))

    # load model
    lenet_cached = LeNet()
    lenet_cached.load_state_dict(torch.load(model_path))
    # model inference
    op = lenet_cached(im)
    # print predictions
    _, pred = torch.max(op, 1)
    print('Prediction: ', ' '.join('%5s' % classes[pred[j]]
                                   for j in range(4)))
    return lenet_cached

def check_accuracy(net, testloader):
    success = 0
    counter = 0
    with torch.no_grad():
        for data in testloader:
            im, ground_truth = data
            op = net(im)
            _, pred = torch.max(op.data, 1)
            counter += ground_truth.size(0)
            success += (pred == ground_truth).sum().item()
    print('Model accuracy on 10000 images from test dataset: %d %%' \
          % (100 * success / counter))

def check_class_accuracy(net, testloader, classes):
    class_success = list(0. for i in range(10))
    class_counter = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            im, ground_truth = data
            op = net(im)
            _, pred = torch.max(op, 1)
            c = (pred == ground_truth).squeeze()
            for i in range(10000):
                ground_truth_curr = ground_truth[i]
                class_success[ground_truth_curr] += c[i].item()
                class_counter[ground_truth_curr] += 1
    for i in range(10):
        print('Model accuracy for class %5s : %2d %%' % (
            classes[i], 100 * class_success[i] / class_counter[i]))
