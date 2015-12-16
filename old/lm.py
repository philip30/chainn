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
def_embed      = 300
def_hidden     = 300
def_batchsize  = 10
def_epoch      = 20
def_lr         = 1.0
grad_clip      = 5
bp_len         = 35

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
    data, batched_data = load_data(args.train, vocab, args.batch_size)
    dev , batched_dev  = load_data(args.dev, vocab, 1)
    test, batched_test = load_data(args.test, vocab, 1)
    model = init_model(input_size = len(vocab),
            embed_size   = args.embed_size,
            hidden_size  = args.hidden_size,
            output_size  = len(vocab))
    optimizer = optimizers.SGD(lr=args.lr)
    
    # Begin Training
    UF.init_model_parameters(model)
    model = UF.convert_to_GPU(USE_GPU, model)
    optimizer.setup(model)
    
    batchsize  = args.batch_size
    epoch      = args.epoch
    accum_loss = Variable(xp.zeros((), dtype=np.float32))
    counter    = 0
    # For each epoch..
    for ep in range(epoch):
        UF.trace("Training Epoch %d" % ep)
        total_tokens = 0
        log_ppl      = 0.0
        
        # For each batch, do forward & backward computations
        for i, batch in enumerate(batched_data):
            loss, nwords  = forward(model, batch)
            accum_loss   += loss
            log_ppl      += loss.data.reshape(())
            # Tracing...
            total_tokens += nwords
#            UF.trace('  %d/%d = %.5f' % (min(i*batchsize, len(data)), len(data), loss.data.reshape(())*batchsize))
            # Counting
            if (counter+1) % bp_len == 0:
                optimizer.zero_grads()
                accum_loss.backward()
                accum_loss.unchain_backward()
                accum_loss = Variable(xp.zeros((), dtype=np.float32))
                
                optimizer.clip_grads(grad_clip)
                optimizer.update()
            counter += 1
        # Counting Perplexity
        log_ppl /= total_tokens
        UF.trace("  PPL (Train)  = %.10f" % math.exp(UF.to_cpu(USE_GPU, log_ppl)))
        dev_ppl = evaluate(model, batched_dev)
        UF.trace("  PPL (Dev)    = %.10f" % math.exp(UF.to_cpu(USE_GPU, dev_ppl)))

        # Reducing learning rate
        if ep > 6:
            optimizer.lr /= 1.2
            UF.trace("Reducing LR:", optimizer.lr)

    # Begin Testing
    UF.trace("Begin Testing...")
    test_ppl = evaluate(model, batched_test)
    UF.trace("  log(PPL) = %.10f" % test_ppl)
    UF.trace("  PPL      = %.10f" % math.exp(UF.to_cpu(USE_GPU, test_ppl)))

"""
Evaluation
"""
def evaluate(model, batched_data):
    total_tokens = 0
    log_ppl      = 0
    for i, test_batch in enumerate(batched_data):
        loss, tokens  = forward(model, test_batch)
        log_ppl      += loss.data.reshape(())
        total_tokens += tokens
    log_ppl /= total_tokens
    return log_ppl

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

def make_initial_state(batch_size, train=True):
    if USE_LSTM:
        return { state_name : Variable(xp.zeros((batch_size, HIDDEN_SIZE),dtype=np.float32),volatile=not train)
            for state_name in ("c1","h1","c2","h2")}
    else:
        return Variable(xp.zeros((batch_size, HIDDEN_SIZE),dtype=np.float32),volatile=not train)

"""
Forward Computation
"""
def forward(model, sents, train=True):
    loss    = 0
    row_len = len(sents)
    col_len = len(sents[0])
    state     = make_initial_state(row_len, train=True)
    for i in range(col_len-1):
        word      = xp.asarray([sents[j][i] for j in range(row_len)]).astype(np.int32)
        next_word = xp.asarray([sents[j][i+1] for j in range(row_len)]).astype(np.int32)
        state, new_loss = forward_one_step(model, state, word, next_word, train=train)
        loss    += new_loss
    return loss, col_len

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
    h1_in  = model.l1_x(F.tanh(h0)) + model.l1_h(state["h1"])
    c1, h1 = F.lstm(state["c1"], h1_in)
    h2_in  = model.l2_x(F.tanh(h1)) + model.l2_h(state["h2"])
    c2, h2 = F.lstm(state["c2"], h2_in)
    y      = model.l3(F.tanh(h2))
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

def load_data(input_data, vocab, batch_size):
    data = []
    # Reading in the data
    with open(input_data,"r") as finput:
        for line in finput:
            sent    = line.strip().lower()
            words   = load_sent(sent.split(), vocab)
            data.append(words)
    data += data[0:batch_size - len(data)%batch_size]
    # Splitting into batch
    batched = []
    for i in range(0, len(data), batch_size):
        this_batch = data[i:i+batch_size]
        batch      = []
        max_len    = max(len(x) for x in this_batch)
        for j in range(len(this_batch)):
            batch.append(this_batch[j] + [vocab["</s>"]] * (max_len - len(this_batch[j])))
        batched.append(batch)

    return data, batched

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
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--hidden_size", type=int, default=def_hidden)
    parser.add_argument("--embed_size", type=int, default=def_embed)
    parser.add_argument("--batch_size", type=int, default=def_batchsize)
    parser.add_argument("--epoch", type=int, default=def_epoch)
    parser.add_argument("--lr", type=float, default=def_lr)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--model",choices=["lstm","rnn"], default="lstm")
    return parser.parse_args()

if __name__ == "__main__":
    main()

