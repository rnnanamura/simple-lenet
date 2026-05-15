import os
import random
import numpy as np
from tqdm import tqdm
from string import punctuation
from collections import Counter
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from typing import cast
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

def load_dataset_from_hf(dataset_name, tokenizer_name, b_size, device, pad_token = "<pad>", unknown_token = "<unk>"):
    dataset = load_dataset(dataset_name)

    # 1. Tokenizer (this part still works fine in modern torchtext)
    tokenizer = get_tokenizer(tokenizer_name)

    # 2. Build vocab from training split
    counts = Counter(
        token
        for example in dataset["train"]
        for token in tokenizer(example["text"])
    )
    specials = [unknown_token, pad_token]
    stoi = {tok: i for i, tok in enumerate(specials)}
    for tok, _ in counts.most_common():
        if tok not in stoi:
            stoi[tok] = len(stoi)
    unk_idx = stoi[unknown_token]
    vocab = type("Vocab", (), {
        "stoi": stoi,
        "__getitem__": lambda self, t: stoi.get(t, unk_idx),
        "__call__": lambda self, tokens: [stoi.get(t, unk_idx) for t in tokens],
        "__len__": lambda self: len(stoi),
    })()

    # 3. Numericalize + get lengths manually
    def process(example):
        tokens = tokenizer(example["text"])
        ids = vocab(tokens)
        return {"input_ids": ids, "length": len(tokens)}

    dataset = dataset.map(process)
    dataset.set_format(type=None, columns=["input_ids", "length", "label"])

    # 3. Split train → train/valid
    train_valid = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_dataset = train_valid["train"]
    valid_dataset = train_valid["test"]
    test_dataset = dataset["test"]

    # 4. Collate (replaces BucketIterator)
    def collate_fn(batch):
        batch = sorted(batch, key=lambda x: x["length"], reverse=True)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["input_ids"]) for x in batch],
            batch_first=False, padding_value=vocab[pad_token]
        )
        lengths = torch.tensor([x["length"] for x in batch])
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.float)
        return input_ids.to(device), lengths.to(device), labels.to(device)

    # 5. DataLoaders (replaces BucketIterator.splits)
    train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=b_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader, vocab, tokenizer

def cuda_pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    lengths = torch.as_tensor(lengths, dtype=torch.int64)
    lengths = lengths.cpu()
    sorted_indices = None
    if not enforce_sorted:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = \
    torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first)
    return PackedSequence(data, batch_sizes, sorted_indices)

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
def train(model, data_iterator, optim, loss_func, device):
    loss = 0
    accuracy = 0
    model.train()

    # data_iterator.to(device)
    for sequence, sequence_lengths, labels in data_iterator:
        optim.zero_grad()
        preds = model(sequence, sequence_lengths).squeeze(1)

        loss_curr = loss_func(preds, labels)
        accuracy_curr = accuracy_metric(preds, labels)

        loss_curr.backward()
        optim.step()

        loss += loss_curr.item()
        accuracy += accuracy_curr.item()

    return loss / len(data_iterator), accuracy / len(data_iterator)


# %%
def validate(model, data, loss_func, device):
    loss = 0
    accuracy = 0
    model.eval()

    #data.to(device)
    with torch.no_grad():
        for sequence, sequence_lengths, labels in data:
            preds = model(sequence, sequence_lengths).squeeze(1)

            loss_curr = loss_func(preds, labels)
            accuracy_curr = accuracy_metric(preds, labels)

            loss += loss_curr.item()
            accuracy += accuracy_curr.item()

    return loss / len(data), accuracy / len(data)


# %%
def sentiment_inference(model, sentence, tokenizer, vocab, device):
    model.eval()

    # text transformations
    tokenized = tokenizer(sentence)
    tokenized = [vocab[t] for t in tokenized]

    # model inference
    model_input = torch.LongTensor(tokenized).to(device)
    model_input = model_input.unsqueeze(1)

    pred = torch.sigmoid(model(model_input))

    return pred.item()

SEQUENCE_LENGTH = 512

def load_dataset_claude(data_dir, split_frac=0.8, batch_size=50):
    reviews, labels = [], []
    for split in ["train", "test"]:
        for label in ["pos", "neg"]:
            folder = os.path.join(data_dir, split, label)
            for fname in tqdm(os.listdir(folder), desc=f"{split}/{label}"):
                if not fname.endswith(".txt"):
                    continue
                with open(os.path.join(folder, fname), encoding="utf8") as f:
                    reviews.append(f.read())
                labels.append(1 if label == "pos" else 0)

    words = " ".join(reviews).lower()
    words = "".join(c for c in words if c not in punctuation).split()
    counts = Counter(words)
    vocab = {w: i + 1 for i, (w, _) in enumerate(counts.most_common())}  # 0 = padding

    tokenized = [[vocab.get(w, 0) for w in r.lower().split()] for r in reviews]
    padded = np.zeros((len(tokenized), SEQUENCE_LENGTH), dtype=int)
    for i, seq in enumerate(tokenized):
        seq = seq[:SEQUENCE_LENGTH]
        padded[i, SEQUENCE_LENGTH - len(seq):] = seq

    indices = list(range(len(padded)))
    random.seed(123)
    random.shuffle(indices)
    split = int(len(indices) * split_frac)
    valid_split = int(split * 0.9)


    train_loader = make_loader(padded, labels, batch_size, indices[:valid_split])
    valid_loader = make_loader(padded, labels, batch_size, indices[valid_split:split])
    test_loader  = make_loader(padded, labels, batch_size, indices[split:])

    return train_loader, valid_loader, test_loader, vocab

def make_loader(padded, labels, batch_size, idx):
    x = torch.tensor(padded[idx], dtype=torch.long)
    y = torch.tensor([labels[i] for i in idx], dtype=torch.float)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
