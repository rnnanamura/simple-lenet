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
from helper import cuda_pack_padded_sequence


# %%
class LSTM(nn.Module):
    def __init__(self, vocabulary_size, embedding_dimension, hidden_dimension, output_dimension, dropout, pad_index):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_dimension, padding_idx=pad_index)
        self.lstm_layer = nn.LSTM(embedding_dimension,
                                  hidden_dimension,
                                  num_layers=1,
                                  bidirectional=True,
                                  dropout=dropout)
        self.fc_layer = nn.Linear(hidden_dimension * 2, output_dimension)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, sequence, sequence_lengths=None):
        if sequence_lengths is None:
            sequence_lengths = torch.LongTensor([len(sequence)])

        # sequence := (sequence_length, batch_size)
        embedded_output = self.dropout_layer(self.embedding_layer(sequence))

        # embedded_output := (sequence_length, batch_size, embedding_dimension)
        if torch.cuda.is_available():
            packed_embedded_output = cuda_pack_padded_sequence(embedded_output, sequence_lengths)
        else:
            packed_embedded_output = nn.utils.rnn.pack_padded_sequence(embedded_output, sequence_lengths)

        packed_output, (hidden_state, cell_state) = self.lstm_layer(packed_embedded_output)
        # hidden_state := (num_layers * num_directions, batch_size, hidden_dimension)
        # cell_state := (num_layers * num_directions, batch_size, hidden_dimension)

        op, op_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # op := (sequence_length, batch_size, hidden_dimension * num_directions)

        hidden_output = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)
        # hidden_output := (batch_size, hidden_dimension * num_directions)

        return self.fc_layer(hidden_output)


