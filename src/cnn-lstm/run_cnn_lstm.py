"""CNN-LSTM

Usage:
    run_cnn_lstm.py [--train | --predict]
    run_cnn_lstm.py (-h | --help)
    run_cnn_lstm.py (-v | --version)

Options:
    --train  train the model.
    --predict  predict the caption for an image.
"""
import os
import nltk
import pickle
from docopt import docopt
import numpy as np
from PIL import Image
from collections import Counter
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

from helper import build_vocabulary, reshape_images, get_loader, load_image
from models import CNNModel, LSTMModel


def main():
    args = docopt(__doc__)
    nltk.download('punkt_tab')

    # %%
    if args['--train']:
        train()

    # predict
    # %%
    if args['--predict']:
        image_file_path = 'sample.jpg'
        predict(image_file_path)


def predict(image_file_path):

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open('data_dir/vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)

    # Build models
    encoder_model = CNNModel(256).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder_model = LSTMModel(256, 512, len(vocabulary), 1)
    encoder_model = encoder_model.to(device)
    decoder_model = decoder_model.to(device)

    # Load the trained model parameters
    encoder_model.load_state_dict(torch.load('models_dir/encoder-2-3000.ckpt'))
    decoder_model.load_state_dict(torch.load('models_dir/decoder-2-3000.ckpt'))

    # Prepare an image
    img = load_image(image_file_path, transform)
    img_tensor = img.to(device)

    # Generate an caption from the image
    feat = encoder_model(img_tensor)
    sampled_indices = decoder_model.sample(feat)
    sampled_indices = sampled_indices[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    predicted_caption = []
    for token_index in sampled_indices:
        word = vocabulary.i2w[token_index]
        predicted_caption.append(word)
        if word == '<end>':
            break
    predicted_sentence = ' '.join(predicted_caption)

    # Print out the image and the generated caption
    print(predicted_sentence)
    img = Image.open(image_file_path)
    plt.imshow(np.asarray(img))


def train():
    vocab = build_vocabulary(json='data_dir/annotations/captions_train2014.json', threshold=4)
    vocab_path = './data_dir/vocabulary.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))
    image_path = './data_dir/train2014/'
    output_path = './data_dir/resized_images/'
    image_shape = [256, 256]
    reshape_images(image_path, output_path, image_shape)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device is {device}")

    # Create model directory
    if not os.path.exists('models_dir/'):
        os.makedirs('models_dir/')

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open('data_dir/vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)

    # Build data loader
    custom_data_loader = get_loader('data_dir/resized_images', 'data_dir/annotations/captions_train2014.json',
                                    vocabulary,
                                    transform, 128,
                                    shuffle=True)

    # Build the models
    encoder_model = CNNModel(256).to(device)
    decoder_model = LSTMModel(256, 512, len(vocabulary), 1).to(device)

    # Loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    parameters = list(decoder_model.parameters()) + list(encoder_model.linear_layer.parameters()) + list(
        encoder_model.batch_norm.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.001)

    # Train the models
    total_num_steps = len(custom_data_loader)
    for epoch in range(5):
        for i, (imgs, caps, lens) in enumerate(custom_data_loader):

            # Set mini-batch dataset
            imgs = imgs.to(device)
            caps = caps.to(device)
            tgts = pack_padded_sequence(caps, lens, batch_first=True)[0]

            # Forward, backward and optimize
            feats = encoder_model(imgs)
            outputs = decoder_model(feats, caps, lens)
            loss = loss_criterion(outputs, tgts)
            decoder_model.zero_grad()
            encoder_model.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, 5, i, total_num_steps, loss.item(), np.exp(loss.item())))

                # Save the model checkpoints
            if (i + 1) % 1000 == 0:
                torch.save(decoder_model.state_dict(), os.path.join(
                    'models_dir/', 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                torch.save(encoder_model.state_dict(), os.path.join(
                    'models_dir/', 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))


if __name__ == "__main__":
    main()
