"""RNN

Usage:
    run_rnn.py [--load]
    run_rnn.py (-h | --help)
    run_rnn.py (-v | --version)

Options:
    --load  load data.
"""


#%%
import os
import time
import numpy as np
from docopt import docopt
from tqdm import tqdm
from string import punctuation
from collections import Counter
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from helper import load_data, pad_sequence, train, validate, sentiment_inference
from models import RNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.use_deterministic_algorithms(True)


def main():
    args = docopt(__doc__)

    (review_list, label_list) = load_data()
    # %%
    #  pre-processing review text
    review_list = [review.lower() for review in review_list]
    review_list = [''.join([letter for letter in review if letter not in punctuation]) for review in tqdm(review_list)]

    # accumulate all review texts together
    reviews_blob = ' '.join(review_list)

    # generate list of all words of all reviews
    review_words = reviews_blob.split()

    # get the word counts
    count_words = Counter(review_words)

    # sort words as per counts (decreasing order)
    total_review_words = len(review_words)
    sorted_review_words = count_words.most_common(total_review_words)

    print(sorted_review_words[:10])
    # %%
    # create word to integer (token) dictionary in order to encode text as numbers
    vocab_to_token = {word: idx + 1 for idx, (word, count) in enumerate(sorted_review_words)}
    print(list(vocab_to_token.items())[:10])
    # %%
    reviews_tokenized = []
    for review in review_list:
        word_to_token = [vocab_to_token[word] for word in review.split()]
        reviews_tokenized.append(word_to_token)
    print(review_list[0])
    print()
    print(reviews_tokenized[0])

    # %%
    # encode sentiments as 0 or 1
    encoded_label_list = [1 if label == 'pos' else 0 for label in label_list]

    reviews_len = [len(review) for review in reviews_tokenized]

    reviews_tokenized = [reviews_tokenized[i] for i, l in enumerate(reviews_len) if l > 0]
    encoded_label_list = np.array([encoded_label_list[i] for i, l in enumerate(reviews_len) if l > 0], dtype='float32')

    sequence_length = 512
    padded_reviews = pad_sequence(reviews_tokenized=reviews_tokenized, sequence_length=sequence_length)

    plt.hist(reviews_len)
    plt.show()

    #%%
    train_val_split = 0.75
    train_X = padded_reviews[:int(train_val_split*len(padded_reviews))]
    train_y = encoded_label_list[:int(train_val_split*len(padded_reviews))]
    validation_X = padded_reviews[int(train_val_split*len(padded_reviews)):]
    validation_y = encoded_label_list[int(train_val_split*len(padded_reviews)):]

    # %%
    # generate torch datasets
    train_dataset = TensorDataset(torch.from_numpy(train_X).to(device), torch.from_numpy(train_y).to(device))
    validation_dataset = TensorDataset(torch.from_numpy(validation_X).to(device),
                                       torch.from_numpy(validation_y).to(device))

    batch_size = 32
    # torch dataloaders (shuffle data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    # %%
    # get a batch of train data
    train_data_iter = iter(train_dataloader)
    X_example, y_example = next(train_data_iter)
    print('Example Input size: ', X_example.size())  # batch_size, seq_length
    print('Example Input:\n', X_example)
    print()
    print('Example Output size: ', y_example.size())  # batch_size
    print('Example Output:\n', y_example)

    input_dimension = len(vocab_to_token)+1 # +1 to account for padding
    embedding_dimension = 100
    hidden_dimension = 32
    output_dimension = 1

    rnn_model = RNN(input_dimension, embedding_dimension, hidden_dimension, output_dimension)

    optim = torch.optim.Adam(rnn_model.parameters())
    loss_func = nn.BCEWithLogitsLoss()

    rnn_model = rnn_model.to(device)
    loss_func = loss_func.to(device)


    # %%
    num_epochs = 10
    best_validation_loss = float('inf')

    for ep in range(num_epochs):

        time_start = time.time()

        training_loss, train_accuracy = train(rnn_model, train_dataloader, optim, loss_func)
        validation_loss, validation_accuracy = validate(rnn_model, validation_dataloader, loss_func)

        time_end = time.time()
        time_delta = time_end - time_start

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(rnn_model.state_dict(), 'rnn_model.pt')
    
        print(f'epoch number: {ep + 1} | time elapsed: {time_delta}s')
        print(f'training loss: {training_loss:.3f} | training accuracy: {train_accuracy * 100:.2f}%')
        print(f'validation loss: {validation_loss:.3f} |  validation accuracy: {validation_accuracy * 100:.2f}%')
        print()

    # %%
    print(sentiment_inference(rnn_model, "This film is horrible", vocab_to_token))
    print(sentiment_inference(rnn_model, "Director tried too hard but this film is bad", vocab_to_token))
    print(sentiment_inference(rnn_model, "This film will be houseful for weeks", vocab_to_token))
    print(sentiment_inference(rnn_model, "I just really loved the movie", vocab_to_token))

if __name__ == '__main__':
    main()