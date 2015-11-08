#!/usr/bin/env python
# This program is used to train multi-labels classifier recursive neural network (using chainer)
# To use this program is very simple:
# $ python3 rnn.py --train [DATA] --test [TEST]

import sys
import chainer.functions as F
import util.functions as UF
import argparse
import numpy as np
from collections import defaultdict
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils

batchsize = 10
epoch     = 30 
xp        = None

def main():
    global xp
    args                   = parse_args()
    x_ids                  = defaultdict(lambda:len(x_ids))
    y_ids                  = defaultdict(lambda:len(y_ids))
    init_wrapper(not args.use_cpu)
    data, target           = load_data(args.train, x_ids, y_ids)
    test_data, test_target = load_data(args.test, x_ids, y_ids)
    model = init_model(input_size = args.input_size,
            embed_size   = args.embed_size,
            hidden_size  = args.hidden_size,
            output_size  = len(y_ids))
    optimizer         = optimizers.SGD()
  
    # Begin Training
    UF.init_model_parameters(model)
    model = UF.convert_to_GPU(not args.use_cpu, model)
    optimizer.setup(model)
    prev_acc = 0
    for ep in range(epoch):
        UF.trace("Training Epoch %d" % ep)
        epoch_acc = 0
        total     = 0
        for i in range(0, len(data), batchsize):
            x_batch = data[i: i+batchsize]
            y_batch = target[i : i+batchsize]
            optimizer.zero_grads()
            loss, accuracy = forward(model, x_batch, y_batch, args.hidden_size)
            loss.backward()
            optimizer.update()
            # Counting epoch accuracy
            epoch_acc += 100 * accuracy.data
            total     += 1
        epoch_acc /= total
        if prev_acc > epoch_acc:
            optimizer.lr *= 0.9
            UF.trace("Reducing LR:", optimizer.lr)
        prev_acc = epoch_acc
        UF.trace("Epoch Accuracy: %.2f" % (epoch_acc))
    
    # Begin Testing
    sum_loss, sum_accuracy = 0, 0
    for i in range(0, len(test_data), batchsize):
        x_batch = test_data[i : i+batchsize]
        y_batch = test_target[i : i+batchsize]
        loss, accuracy = forward(model, x_batch, y_batch, args.hidden_size)
        sum_loss      += loss.data * batchsize
        sum_accuracy  += accuracy.data * batchsize
    mean_loss     = sum_loss / len(test_data)
    mean_accuracy = sum_accuracy / len(test_data)
    print("Mean Loss", mean_loss)
    print("Mean Accuracy", mean_accuracy)

def init_wrapper(use_gpu):
    global xp
    xp = UF.select_wrapper(use_gpu)

def forward_one_step(model, h, cur_word, label, volatile=False):
    word = Variable(cur_word)
    t    = Variable(label, volatile=volatile)
    x    = F.tanh(model.embed(word))
    h    = F.tanh(model.x_to_h(x) + model.h_to_h(h))
    y    = model.h_to_y(h)
    loss = F.softmax_cross_entropy(y, t)
    accuracy = F.accuracy(y,t)
    return h, loss, accuracy

def forward(model, x_batch, y_batch, hidden_size, volatile=False):
    loss    = 0
    correct = 0
    total   = 0
    for x_list, y_list in zip(x_batch, y_batch):
        h   = Variable(xp.zeros((1,hidden_size), dtype=np.float32), volatile=False)
        for i in range(len(x_list)):
            h, new_loss, score = forward_one_step(model, h, x_list[i:i+1], y_list[i:i+1], volatile=volatile)
            loss    += new_loss
            correct += score
            total   += 1

    accuracy = correct / total
    return loss, accuracy


def init_model(input_size, embed_size, hidden_size, output_size):
    model = FunctionSet(
        embed  = F.EmbedID(input_size, embed_size),
        x_to_h = F.Linear(embed_size, hidden_size),
        h_to_h = F.Linear(hidden_size, hidden_size),
        h_to_y = F.Linear(hidden_size, output_size)
    )
    return model

def load_data(input_data, x_ids, y_ids):
    x_ids  = load_dict(x_ids)
    y_ids  = load_dict(y_ids)
    holder = []
    # Reading in the data
    with open(input_data,"r") as finput:
        for line in finput:
            sent = line.strip().lower()
            words, labels = load_sent(sent.split(), x_ids, y_ids)
            holder.append((words,labels))
    return convert_to_array(holder)
    
def convert_to_array(holder):
    # Convert to appropriate data structure
    data, target   = [], []
    for i, (words, tags) in enumerate(holder):
        sent_array = []
        tags_array = []
        for j, (w, t) in enumerate(zip(words, tags)):
            sent_array.append(w)
            tags_array.append(t)
            
        data.append(xp.array(sent_array).astype(np.int32))
        target.append(xp.array(tags_array).astype(np.int32))
    return data, target

def load_dict(ids):
    return ids

def load_sent(tokens, x_ids, y_ids):
    words  = []
    labels = []
    for word in tokens:
        word, tag = word.split("_")
        words.append(x_ids[word])
        labels.append(y_ids[tag])
    return words, labels

def parse_args():
    parser = argparse.ArgumentParser(description="A program to run Recursive Neural Network classifier using chainner")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--input_size", type=int, default=4000)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--embed_size", type=int, default=100)
    parser.add_argument("--use_cpu", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    main()
