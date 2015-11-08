#!/usr/bin/env python
# This program is used to train a language model (rnn/lstm) (using chainer)
# To use this program is very simple:
# $ python3 lm.py --train [DATA] --test [TEST] [--model [lstm/rnn]]

import sys
import chainer.functions as F
import util.functions as UF
import argparse
import math
import numpy as np
from collections import defaultdict
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils

# Default parameters
def_embed     = 300
def_hidden    = 300
def_batchsize = 256
def_epoch     = 20
def_lr        = 0.1

# Global vocab
dictionary = {}
xp         = None

# Program State
USE_CPU     = True
USE_LSTM    = True
HIDDEN_SIZE = def_hidden

def main():
    args      = parse_args()
    init_program_state(args)
    vocab     = make_vocab()
    data      = load_data(args.train, vocab)
    test_data = load_data(args.test, vocab)
    model = init_model(input_size = len(vocab),
            embed_size   = args.embed_size,
            hidden_size  = args.hidden_size,
            output_size  = len(vocab))
    optimizer = optimizers.SGD(lr=args.lr)
    
    # Begin Training
    UF.init_model_parameters(model)
    model = UF.convert_to_GPU(USE_GPU, model)
    optimizer.setup(model)
    
    batchsize = args.batch_size
    epoch     = args.epoch
    # For each epoch..
    for ep in range(epoch):
        UF.trace("Training Epoch %d" % ep)
        total_tokens = 0
        log_ppl      = 0.0
        # For each batch, do forward & backward computation
        for i in range(0, len(data), batchsize):
            this_batch = data[i: i+batchsize]
            optimizer.zero_grads()
            loss, tokens  = forward(model, this_batch)
            log_ppl      += loss.data.reshape(()) * batchsize
            loss.backward()
            optimizer.update()
            # Tracing...
            total_tokens += tokens
            UF.trace('  %d/%d = %.5f' % (min(i+batchsize, len(data)), len(data), loss.data.reshape(())*batchsize))
        # Counting Perplexity
        log_ppl /= total_tokens
        UF.trace("  log(PPL) = %.10f" % log_ppl)
        UF.trace("  PPL      = %.10f" % math.exp(UF.to_cpu(USE_GPU, log_ppl)))
        # Reducing learning rate
        if ep > 6:
            optimizer.lr /= 1.2
            UF.trace("Reducing LR:", optimizer.lr)

    # Begin Testing
    UF.trace("Begin Testing...")
    total_tokens = 0
    log_ppl      = 0
    for i in range(len(test_data)):
        loss, tokens  = forward(model, test_data[i:i+1], train=False)
        log_ppl      += loss.data.reshape(())
        total_tokens += tokens
        UF.trace('  %d/%d ' % (i, len(test_data)))
    log_ppl /= total_tokens
    UF.trace("  log(PPL) = %.10f" % log_ppl)
    UF.trace("  PPL      = %.10f" % math.exp(UF.to_cpu(USE_GPU, log_ppl)))

"""
Model initialization
"""
def init_model(input_size, embed_size, hidden_size, output_size):
    init = init_model_lstm if USE_LSTM else init_model_rnn
    return init(input_size, embed_size, hidden_size, output_size)

def init_model_lstm(input_size, embed_size, hidden_size, output_size):
    model = FunctionSet(
        embed  = F.EmbedID(input_size, embed_size),
        l1_x   = F.Linear(embed_size, 4 * hidden_size),
        l1_h   = F.Linear(hidden_size, 4 * hidden_size),
        l2_x   = F.Linear(hidden_size, 4 * hidden_size),
        l2_h   = F.Linear(hidden_size, 4 * hidden_size),
        l3     = F.Linear(hidden_size, output_size)
    )
    return model

def init_model_rnn(input_size, embed_size, hidden_size, output_size):
    model = FunctionSet(
        embed  = F.EmbedID(input_size, embed_size),
        x_to_h = F.Linear(embed_size, hidden_size),
        h_to_h = F.Linear(hidden_size, hidden_size),
        h_to_y = F.Linear(hidden_size, output_size)
    )
    return model

def make_initial_state(train=True):
    if USE_LSTM:
        return { state_name : Variable(xp.zeros((1, HIDDEN_SIZE),dtype=np.float32),volatile=not train)
            for state_name in ("c1","h1","c2","h2")}
    else:
        return Variable(xp.zeros((1,HIDDEN_SIZE),dtype=np.float32),volatile=not train)

"""
Forward Computation
"""
def forward(model, sents, train=True):
    loss    = 0
    tokens  = 0
    for sent in sents:
        state   = make_initial_state(train=train)
        for i in range(len(sent)-1):
            word            = sent[i:i+1]
            next_word       = sent[i+1:i+2]
            state, new_loss = forward_one_step(model, state, word, next_word, train=train)
            loss    += new_loss
            tokens  += 1
    return loss, tokens

def forward_one_step(model, state, cur_word, next_word, train=False):
    if USE_LSTM:
        return forward_one_step_lstm(model, state, cur_word, next_word, train)
    else:
        return forward_one_step_rnn(model, state, cur_word, next_word, train)

def forward_one_step_rnn(model, h, cur_word, next_word, train=True):
    word = Variable(cur_word, volatile=not train)
    t    = Variable(next_word, volatile=not train)
    x    = F.tanh(model.embed(word))
    h    = F.tanh(model.x_to_h(x) + model.h_to_h(h))
    y    = model.h_to_y(h)
    loss = F.softmax_cross_entropy(y, t)
    return h, loss

def forward_one_step_lstm(model, state, cur_word, next_word, train=True):
    x      = Variable(cur_word, volatile=not train)
    t      = Variable(next_word, volatile=not train)
    h0     = model.embed(x)
    h1_in  = model.l1_x(F.dropout(h0, train=train)) + model.l1_h(state["h1"])
    c1, h1 = F.lstm(state["c1"], h1_in)
    h2_in  = model.l2_x(F.dropout(h1, train=train)) + model.l2_h(state["h2"])
    c2, h2 = F.lstm(state["c2"], h2_in)
    y      = model.l3(F.dropout(h2, train=train))
    state  = {"c1": c1, "h1": h1, "c2": c2, "h2":h2}
    loss   = F.softmax_cross_entropy(y, t)
    return state, loss

"""
Utility functions
"""
def init_program_state(args):
    global xp, USE_LSTM, USE_GPU, HIDDEN_SIZE
    xp          = UF.select_wrapper(not args.use_cpu)
    HIDDEN_SIZE = args.hidden_size
    USE_GPU     = not args.use_cpu
    USE_LSTM    = args.model == "lstm"

def load_data(input_data, vocab):
    data = []
    # Reading in the data
    with open(input_data,"r") as finput:
        for line in finput:
            sent    = line.strip().lower()
            words   = load_sent(sent.split(), vocab)
            data.append(xp.array(words).astype(np.int32))
    return data

def make_vocab():
    vocab = defaultdict(lambda:len(vocab))
    vocab["<s>"] = 0
    vocab["</s>"] = 1
    dictionary[0] = "<s>"
    dictionary[1] = "</s>"
    return vocab

def load_sent(tokens, vocab):
    ret = [vocab["<s>"]] + list(map(lambda x: vocab[x], tokens)) + [vocab["</s>"]]
    for tok in tokens:
        dictionary[vocab[tok]] = tok
    return ret

"""
Arguments
"""
def parse_args():
    parser = argparse.ArgumentParser(description="A program to run Recursive Neural Network classifier using chainner")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--hidden_size", type=int, default=def_hidden)
    parser.add_argument("--embed_size", type=int, default=def_embed)
    parser.add_argument("--batch_size", type=int, default=def_batchsize)
    parser.add_argument("--epoch", type=int, default=def_epoch)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--model",choices=["lstm","rnn"], default="lstm")
    return parser.parse_args()

if __name__ == "__main__":
    main()

