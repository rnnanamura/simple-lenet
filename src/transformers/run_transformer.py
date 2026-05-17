"""Transformers

Usage:
    run_transformers.py [--train]
    run_transformers.py (-h | --help)
    run_transformers.py (-v | --version)

Options:
    --train  train the model.
"""

#%%
import math
import time

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

# from torchtext.datasets import PennTreebank
#from torchtext.data.utils import get_tokenizer
#from torchtext.vocab import build_vocab_from_iterator

from datasets import Dataset, DatasetDict

from docopt import docopt
from helper import process_data, gen_batches, train_model, eval_model, load_ptb_split, yield_tokens, build_vocab
from models import Transformer, PosEnc
torch.use_deterministic_algorithms(True)


def main():
    args = docopt(__doc__)
    #%%
    # tr_iter = PennTreebank(split='train')
    # tr_iter, val_iter, te_iter = PennTreebank()
    # Wall Street Journal portion (standard PTB split)
    #wsj_dataset = load_dataset("ptb-text-only/ptb_text_only")
    #train = wsj_dataset["train"]
    #valid = wsj_dataset["validation"]
    #test = wsj_dataset["test"]

    dataset = DatasetDict({
        "train": load_ptb_split("https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt"),
        "validation": load_ptb_split("https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt"),
        "test": load_ptb_split("https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"),
    })
    train = dataset["train"]
    validation = dataset["validation"]
    test = dataset["test"]


    #tkzer = get_tokenizer('basic_english')
    #vocabulary = build_vocab_from_iterator(
    #    yield_tokens(dataset["train"]))
    #vocabulary.set_default_index(vocabulary['<unk>'])

    customVocabulary = build_vocab(dataset["train"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_text = process_data(train, customVocabulary)
    validation_text = process_data(validation, customVocabulary)
    testing_text = process_data(test, customVocabulary)


    training_batch_size = 32
    evaluation_batch_size = 16

    training_data = gen_batches(training_text, training_batch_size, device)
    validation_data = gen_batches(validation_text, evaluation_batch_size, device)
    testing_data = gen_batches(testing_text, evaluation_batch_size, device)

    # %%
    num_tokens = len(customVocabulary.idx2word)  # vocabulary size
    embedding_size = 256  # dimension of embedding layer
    num_hidden_params = 256  # transformer encoder's hidden (feed forward) layer dimension
    num_layers = 2  # num of transformer encoder layers within transformer encoder
    num_heads = 2  # num of heads in (multi head) attention models
    dropout = 0.25  # value (fraction) of dropout
    loss_func = nn.CrossEntropyLoss()
    lrate = 4.0  # learning rate
    transformer_model = Transformer(num_tokens, embedding_size, num_heads, num_hidden_params, num_layers,
                                    dropout).to(device)
    optim_module = torch.optim.SGD(transformer_model.parameters(), lr=lrate)
    sched_module = torch.optim.lr_scheduler.StepLR(optim_module, 1.0, gamma=0.88)

    #%%
    min_validation_loss = float("inf")
    eps = 5
    best_model_so_far = None

    for ep in range(1, eps + 1):
        ep_time_start = time.time()
        train_model(transformer_model, training_data, optim_module, num_tokens, loss_func, sched_module,ep, device)
        validation_loss = eval_model(transformer_model, validation_data, num_tokens, loss_func, device)
        print()
        print(f"epoch {ep:}, validation loss {validation_loss:.2f}, validation perplexity {math.exp(validation_loss):.2f}")
        print()

        if validation_loss < min_validation_loss:
            min_validation_loss = validation_loss
            best_model_so_far = transformer_model
        sched_module.step()

    # %%
    testing_loss = eval_model(best_model_so_far, testing_data, num_tokens, loss_func, device)
    print(f"testing loss {testing_loss:.2f}, testing perplexity {math.exp(testing_loss):.2f}")

if __name__ == "__main__":
    main()