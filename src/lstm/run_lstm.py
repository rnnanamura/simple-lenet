"""LSTM

Usage:
    run_lstm.py [--train]
    run_lstm.py (-h | --help)
    run_lstm.py (-v | --version)

Options:
    --train  train the model.
"""

import os
import time
import numpy as np
from tqdm import tqdm
from string import punctuation
from collections import Counter
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from docopt import docopt
from helper import load_dataset, load_dataset_from_hf, sentiment_inference, train, accuracy_metric, validate
from models import LSTM

def main():
    args = docopt(__doc__)
    torch.use_deterministic_algorithms(True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pad_token = "<pad>"
    unk_token = "<unk>"
    train_loader, valid_loader, test_loader, vocab, tokenizer = load_dataset_from_hf("stanfordnlp/imdb", "basic_english",
          b_size = 64, device = device,
          pad_token = pad_token,
          unknown_token=unk_token)
    # %%
    MAX_VOCABULARY_SIZE = 25000


    INPUT_DIMENSION = len(vocab)
    EMBEDDING_DIMENSION = 100
    HIDDEN_DIMENSION = 32
    OUTPUT_DIMENSION = 1
    DROPOUT = 0.5
    PAD_INDEX = vocab.stoi[pad_token]

    lstm_model = LSTM(INPUT_DIMENSION,
                      EMBEDDING_DIMENSION,
                      HIDDEN_DIMENSION,
                      OUTPUT_DIMENSION,
                      DROPOUT,
                      PAD_INDEX)
    UNK_INDEX = vocab.stoi[unk_token]

    lstm_model.embedding_layer.weight.data[UNK_INDEX] = torch.zeros(EMBEDDING_DIMENSION)
    lstm_model.embedding_layer.weight.data[PAD_INDEX] = torch.zeros(EMBEDDING_DIMENSION)
    optim = torch.optim.Adam(lstm_model.parameters())
    loss_func = nn.BCEWithLogitsLoss()

    lstm_model = lstm_model.to(device)
    loss_func = loss_func.to(device)

    if args['--train']:
        custom_train(lstm_model, train_loader, valid_loader, optim, loss_func, device)

    # %%
    lstm_model.load_state_dict(torch.load('./lstm_model.pt'))

    test_loss, test_accuracy = validate(lstm_model, test_loader, loss_func, device)

    print(f'test loss: {test_loss:.3f} | test accuracy: {test_accuracy * 100:.2f}%')

    # %%
    print(sentiment_inference(lstm_model, "This film is horrible", tokenizer, vocab, device))
    print(sentiment_inference(lstm_model, "Director tried too hard but this film is bad", tokenizer, vocab, device))
    print(sentiment_inference(lstm_model, "This film will be houseful for weeks", tokenizer, vocab, device))
    print(sentiment_inference(lstm_model, "I just really loved the movie", tokenizer, vocab, device))

def custom_train(lstm_model, train_data_iterator, valid_data_iterator, optim, loss_func,  device):
    # %%
    num_epochs = 10
    best_validation_loss = float('inf')

    for ep in range(num_epochs):

        time_start = time.time()

        training_loss, train_accuracy = train(lstm_model, train_data_iterator, optim, loss_func, device)
        validation_loss, validation_accuracy = validate(lstm_model, valid_data_iterator, loss_func, device)

        time_end = time.time()
        time_delta = time_end - time_start

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(lstm_model.state_dict(), 'lstm_model.pt')

        print(f'epoch number: {ep + 1} | time elapsed: {time_delta}s')
        print(f'training loss: {training_loss:.3f} | training accuracy: {train_accuracy * 100:.2f}%')
        print(f'validation loss: {validation_loss:.3f} |  validation accuracy: {validation_accuracy * 100:.2f}%')
        print()

if __name__ == "__main__":
    main()