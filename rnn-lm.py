#!/usr/bin/env python
# This program is used to train multi-labels classifier recursive neural network (using chainer)
# To use this program is very simple:
# $ python3 rnn.py --train [DATA] --test [TEST]

import sys
import chainer.functions as F
import util.functions as UF
import argparse
import math
import numpy as np
from collections import defaultdict
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils

# Default parameters
USE_GPU       = True
def_input     = 25000
def_embed     = 1024 
def_hidden    = 256
def_batchsize = 10
def_epoch     = 10

# Global vocab
dictionary = {}
xp         = cuda.cupy if USE_GPU else np

def main():
    args  = parse_args()
    vocab = make_vocab()
    data  = load_data(args.train, vocab)
    model = init_model(input_size = len(vocab),
            embed_size   = args.embed_size,
            hidden_size  = args.hidden_size,
            output_size  = len(vocab))
    optimizer = optimizers.SGD()

    if len(vocab) > args.input_size:
        raise Exception("Give higher input size! #word in data: %d vs #size:%d" % (len(vocab),args.input_size))
    
    if USE_GPU:
        cuda.check_cuda_available()
        cuda.get_device(0).use()

    # Begin Training
    optimizer.setup(model)
    prev_acc  = 0.0
    
    batchsize = args.batchsize
    epoch     = args.epoch
    for ep in range(epoch):
        UF.trace("Training Epoch %d" % ep)
        epoch_acc = 0
        total     = 0
        log_ppl   = 0.0
        for i in range(0, len(data), batchsize):
            this_batch = data[i: i+batchsize]
            optimizer.zero_grads()
            loss, accuracy = forward(model, this_batch, args.hidden_size)
            log_ppl       += loss.data.reshape(()) * batchsize
            loss.backward()
            optimizer.update()
            # Counting epoch accuracy
            epoch_acc += 100 * accuracy.data
            total     += 1
            UF.trace('  %d/%d ' % (min(i+batchsize, len(data)), len(data)))
        epoch_acc /= total
#        optimizer.lr *= 0.5
#        UF.trace("Reducing LR:", optimizer.lr)
        prev_acc = epoch_acc
        UF.trace("  log(PPL) = %.10f" % log_ppl)
#        UF.trace("  PPL      = %.10f" % math.exp(log_ppl))
        UF.trace("Epoch Accuracy: %.2f" % (epoch_acc))
    sys.exit(1)

    # Begin Testing
#    sum_loss, sum_accuracy = 0, 0
#    for i in range(0, len(test_data), batchsize):
#        x_batch = test_data[i : i+batchsize]
#        y_batch = test_target[i : i+batchsize]
#        loss, accuracy = forward(model, x_batch, y_batch, args.hidden_size)
#        sum_loss      += loss.data * batchsize
#        sum_accuracy  += accuracy.data * batchsize
#    mean_loss     = sum_loss / len(test_data)
#    mean_accuracy = sum_accuracy / len(test_data)
#    print("Mean Loss", mean_loss)
#    print("Mean Accuracy", mean_accuracy)


def forward_one_step(model, h, cur_word, next_word, volatile=False):
    word = Variable(cur_word, volatile=volatile)
    t    = Variable(next_word, volatile=volatile)
    x    = F.tanh(model.embed(word))
    h    = F.tanh(model.x_to_h(x) + model.h_to_h(h))
    y    = model.h_to_y(h)
    loss = F.softmax_cross_entropy(y, t)
    accuracy = F.accuracy(y,t)
    return h, loss, accuracy

def forward(model, sents, hidden_size, volatile=False):
    loss    = 0
    correct = 0
    total   = 0
    for sent in sents:
        h   = Variable(xp.zeros((1,hidden_size), dtype=np.float32), volatile)
        for i in range(len(sent)-1):
            word      = sent[i:i+1]
            next_word = sent[i+1:i+2]
            h, new_loss, score = forward_one_step(model, h, word, next_word, volatile=volatile)
            if new_loss.data < 10e6:
                loss     += new_loss
            correct  += score
            total += 1
    accuracy = correct / total
    return loss, accuracy


def init_model(input_size, embed_size, hidden_size, output_size):
    model = FunctionSet(
        embed  = F.EmbedID(input_size, embed_size),
        x_to_h = F.Linear(embed_size, hidden_size),
        h_to_h = F.Linear(hidden_size, hidden_size),
        h_to_y = F.Linear(hidden_size, output_size)
    )
    if USE_GPU:
        return model.to_gpu()
    else:
        return model

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

def parse_args():
    parser = argparse.ArgumentParser(description="A program to run Recursive Neural Network classifier using chainner")
    parser.add_argument("--train", type=str, required=True)
#    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--input_size", type=int, default=def_input)
    parser.add_argument("--hidden_size", type=int, default=def_hidden)
    parser.add_argument("--embed_size", type=int, default=def_embed)
    parser.add_argument("--batchsize", type=int, default=def_batchsize)
    parser.add_argument("--epoch", type=int, default=def_epoch)
    return parser.parse_args()

if __name__ == "__main__":
    main()
