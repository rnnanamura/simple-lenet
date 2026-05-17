#%%
import math
import time

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import requests
from collections import Counter
from datasets import Dataset, DatasetDict

# for tokernizer instead of torchtext.data.util.tokenizer
import nltk
nltk.download("punkt_tab")

from nltk.tokenize import word_tokenize

torch.use_deterministic_algorithms(True)

class CustomVocabulary(object):
    def __init__(self, word2idx, idx2word):
        self.word2idx: dict = word2idx
        self.idx2word: dict = idx2word
    def add_word2idx(self, word2idx):
        self.word2idx = word2idx
    def add_idx2word(self, idx2word):
        self.idx2word = idx2word
    def __len__(self):
        return len(self.idx2word)
    def numericalize(self, tokens):
        unk_idx = self.word2idx["<unk>"]
        return [self.word2idx.get(t, unk_idx) for t in tokens]

def load_ptb_split(url):
    response = requests.get(url)
    sentences = [line.strip() for line in response.text.splitlines() if line.strip()]
    return Dataset.from_dict({"sentence": sentences})

def tokenize(text):
    return word_tokenize(text.lower())

def yield_tokens(data):
    for ex in data:
        yield tokenize(ex["sentence"])

def build_vocab(dataset, specials=["<unk>", "<pad>"], min_freq=1):
    counter = Counter()
    for example in dataset:
        counter.update(tokenize(example["sentence"]))

    # specials first, then tokens sorted by frequency
    tokens = specials + [t for t, c in counter.most_common() if c >= min_freq]
    stoi = {token: idx for idx, token in enumerate(tokens)}
    itos = tokens

    return CustomVocabulary(stoi, itos)

def process_data(raw_text, vocabulary):
    numerical_text = [torch.tensor(vocabulary.numericalize(ex["sentence"]), dtype=torch.long) for ex in raw_text]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, numerical_text)))

def gen_batches(text_dataset, batch_size, device):
    num_batches = text_dataset.size(0) // batch_size
    text_dataset = text_dataset[:num_batches * batch_size]
    text_dataset = text_dataset.view(batch_size, num_batches).t().contiguous()
    return text_dataset.to(device)

#%%
max_seq_len = 64
def return_batch(src, k):
    sequence_length = min(max_seq_len, len(src) - 1 - k)
    sequence_data = src[k:k+sequence_length]
    sequence_label = src[k+1:k+1+sequence_length].reshape(-1)
    return sequence_data, sequence_label

def gen_sqr_nxt_mask(size):
    msk = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return msk

#%%
def train_model(transformer_model, training_data, optim_module,
                num_tokens, loss_func, sched_module, ep, device):
    transformer_model.train()
    loss_total = 0.
    time_start = time.time()
    mask_source = gen_sqr_nxt_mask(max_seq_len).to(device)
    num_batches = len(training_data) // max_seq_len
    for b, i in enumerate(range(0, training_data.size(0) - 1, max_seq_len)):
        train_data_batch, train_label_batch = return_batch(training_data, i)
        sequence_length = train_data_batch.size(0)
        if sequence_length != max_seq_len:  # only on last batch
            mask_source = mask_source[:sequence_length, :sequence_length]
        op = transformer_model(train_data_batch, mask_source)
        loss_curr = loss_func(op.view(-1, num_tokens), train_label_batch)
        optim_module.zero_grad()
        loss_curr.backward()
        torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), 0.6)
        optim_module.step()

        loss_total += loss_curr.item()
        interval = 100
        if b % interval == 0 and b > 0:
            loss_interval = loss_total / interval
            time_delta = time.time() - time_start
            print(f"epoch {ep}, {b}/{len(training_data)//max_seq_len} batches, training loss {loss_interval:.2f}, training perplexity {math.exp(loss_interval):.2f}")
            loss_total = 0
            time_start = time.time()

def eval_model(eval_model_obj, eval_data_source, num_tokens, loss_func, device):
    eval_model_obj.eval()
    loss_total = 0.
    mask_source = gen_sqr_nxt_mask(max_seq_len).to(device)
    with torch.no_grad():
        for j in range(0, eval_data_source.size(0) - 1, max_seq_len):
            eval_data, eval_label = return_batch(eval_data_source, j)
            sequence_length = eval_data.size(0)
            if sequence_length != max_seq_len:
                mask_source = mask_source[:sequence_length, :sequence_length]
            op = eval_model_obj(eval_data, mask_source)
            op_flat = op.view(-1, num_tokens)
            loss_total += sequence_length * loss_func(op_flat, eval_label).item()
    return loss_total / (len(eval_data_source) - 1)