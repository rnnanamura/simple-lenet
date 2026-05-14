#%%
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.use_deterministic_algorithms(True)

def load_data():
    review_list = []
    label_list = []
    for label in ['pos', 'neg']:
        for fname in tqdm(os.listdir(f'./aclImdb/train/{label}/')):
            if 'txt' not in fname:
                continue
            with open(os.path.join(f'./aclImdb/train/{label}/', fname), encoding="utf8") as f:
                review_list += [f.read()]
                label_list += [label]
    print('Number of reviews :', len(review_list))
    return review_list, label_list


# %%
def pad_sequence(reviews_tokenized, sequence_length):
    ''' returns the tokenized review sequences padded with 0's or truncated to the sequence_length.
    '''
    padded_reviews = np.zeros((len(reviews_tokenized), sequence_length), dtype=int)

    for idx, review in enumerate(reviews_tokenized):
        review_len = len(review)

        if review_len <= sequence_length:
            zeroes = list(np.zeros(sequence_length - review_len))
            new_sequence = zeroes + review
        elif review_len > sequence_length:
            new_sequence = review[0:sequence_length]

        padded_reviews[idx, :] = np.array(new_sequence)

    return padded_reviews
#%%
def accuracy_metric(predictions, ground_truth):
    """
    Returns 0-1 accuracy for the given set of predictions and ground truth
    """
    # round predictions to either 0 or 1
    rounded_predictions = torch.round(torch.sigmoid(predictions))
    success = (rounded_predictions == ground_truth).float() #convert into float for division
    accuracy = success.sum() / len(success)
    return accuracy


# %%
def train(model, dataloader, optim, loss_func):
    loss = 0
    accuracy = 0
    model.train()

    for sequence, sentiment in dataloader:
        optim.zero_grad()
        preds = model(sequence.T).squeeze()

        loss_curr = loss_func(preds, sentiment)
        accuracy_curr = accuracy_metric(preds, sentiment)

        loss_curr.backward()
        optim.step()

        loss += loss_curr.item()
        accuracy += accuracy_curr.item()

    return loss / len(dataloader), accuracy / len(dataloader)


# %%
def validate(model, dataloader, loss_func):
    loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for sequence, sentiment in dataloader:
            preds = model(sequence.T).squeeze()

            loss_curr = loss_func(preds, sentiment)
            accuracy_curr = accuracy_metric(preds, sentiment)

            loss += loss_curr.item()
            accuracy += accuracy_curr.item()

    return loss / len(dataloader), accuracy / len(dataloader)


# %%
def sentiment_inference(model, sentence, vocab_to_token):
    model.eval()

    # text transformations
    sentence = sentence.lower()
    sentence = ''.join([c for c in sentence if c not in punctuation])
    tokenized = [vocab_to_token.get(token, 0) for token in sentence.split()]
    tokenized = np.pad(tokenized, (512 - len(tokenized), 0), 'constant')

    # model inference
    model_input = torch.LongTensor(tokenized).to(device)
    model_input = model_input.unsqueeze(1)
    pred = torch.sigmoid(model(model_input))

    return pred.item()