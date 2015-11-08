#!/usr/bin/env python
# This program is used to train binary classifier neural network (using chainer)
# To use this program is very simple:
# $ python3 nn.py --train [DATA] --test [TEST]

import sys
import chainer.functions as F
import util.functions as UF
import argparse
import numpy as np
from collections import defaultdict
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils

batchsize = 100
epoch     = 5

def main():
    args              = parse_args()
    data, target, ids = load_data(args.train)
    test_data, test_target, ids = load_data(args.test, ids)
    model             = init_model(input_size = len(ids),
            depth        = args.depth,
            hidden_size  = args.hidden_size,
            output_size  = 2)
    optimizer         = optimizers.SGD()
    
    # Begin Training
    optimizer.setup(model)
    for ep in range(epoch):
        UF.trace("Training Epoch %d" % ep)
        indexes = np.random.permutation(len(data))
        for i in range(0, len(data), batchsize):
            x_batch = data[indexes[i: i+batchsize]]
            y_batch = target[indexes[i : i+batchsize]]
            optimizer.zero_grads()
            loss, accuracy = forward(model,x_batch, y_batch)
            loss.backward()
            optimizer.update()
            UF.trace(accuracy.data)

    # Begin Testing
    sum_loss, sum_accuracy = 0, 0
    for i in range(0, len(test_data), batchsize):
        x_batch         = test_data[i : i+batchsize]
        y_batch         = test_target[i : i+batchsize]
        loss, accuracy  = forward(model, x_batch, y_batch)
        sum_loss       += loss.data * batchsize
        sum_accuracy   += accuracy.data * batchsize
    mean_loss     = sum_loss / len(test_data)
    mean_accuracy = sum_accuracy / len(test_data)
    print("Mean Loss", mean_loss)
    print("Mean Accuracy", mean_accuracy)

def forward(model, x_data, y_data):
    x = Variable(x_data)
    t = Variable(y_data)
    h1 = F.relu(model.l1(x))
    h2 = F.relu(model.l2(h1))
    y = model.l3(h2)
    return F.softmax_cross_entropy(y,t), F.accuracy(y,t)

def init_model(input_size, depth, hidden_size, output_size):
    model = FunctionSet(
        l1 = F.Linear(input_size, hidden_size),
        l2 = F.Linear(hidden_size, hidden_size),
        l3 = F.Linear(hidden_size, output_size)
    )
    return model

def load_data(data, ids=None):
    ids, is_train = load_dict(ids)
    holder = []
    # Reading in the data
    with open(data,"r") as finput:
        for line in finput:
            label, sent = line.strip().lower().split("\t")
            sent        = load_sent(sent.split(), ids, is_train)
            label       = 1 if int(label) == 1 else 0
            holder.append((sent, label))
    # Creating an appropriate data structure
    num, dim = len(holder), len(ids)
    data     = np.zeros(num * dim, dtype = np.uint8).reshape((num, dim))
    target   = np.zeros(num, dtype = np.uint8).reshape((num, ))
    for i, (sent, label) in enumerate(holder):
        for word in sent:
            data[i][word] = 1
        target[i] = label
    return data.astype(np.float32), target.astype(np.int32), ids

def load_dict(ids):
    is_train = True
    if ids is None:
        id_dict = defaultdict(lambda:len(id_dict))
    else:
        id_dict = dict(ids)
        is_train = False
    return id_dict, is_train

def load_sent(words, ids, is_train):
    ret = []
    for word in words:
        if is_train:
            ret.append(ids[word])
        elif word in ids:
            ret.append(ids[word])
        else:
            pass # ignore unknown word
    return ret

def parse_args():
    parser = argparse.ArgumentParser(description="A program to run Neural Network classifier using chainner")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--depth", type=int, default=3)
    return parser.parse_args()

if __name__ == "__main__":
    main()

