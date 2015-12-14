#!/usr/bin/env python
# This program is a tutorial purpose with mnist data
# Usage: mnist.py [MNIST_PCKL]

import six
import sys
import numpy as np

from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from chainn import functions as UF

BATCH  = 100
EPOCH  = 20
HIDDEN = 100
INPUT  = 28 * 28    # 28x28 pixels
OUTPUT = 10         # 0,1,2,3,4,5,6,7,8,9

# Load data
# For generating the pickle, look at https://github.com/pfnet/chainer/blob/master/examples/mnist/data.py
def load_data(data_dir):
    with open(data_dir, "rb") as mnist_pickle:
        mnist = six.moves.cPickle.load(mnist_pickle)
    
    x_all = mnist["data"].astype(np.float32) / 255
    y_all = mnist["target"].astype(np.int32)
    
    assert(len(x_all) == len(y_all))
    return x_all, y_all

# MLP
class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
            l1 = L.Linear(INPUT, HIDDEN),
            l2 = L.Linear(HIDDEN, HIDDEN),
            l3 = L.Linear(HIDDEN, OUTPUT),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

# NOTE that this class is defined similarly in chainer.links.Classifier
#class Classifier(Chain):
#    def __init__(self, predictor):
#        super(classifier, self).__init__(predictor=predictor)
#
#    def __call__(self, x, t):
#        y = self.preditor(x)
#        self.loss = F.softmax_cross_entropy(y,t)
#        self.accuracy = F.accuracy(y, t)
#        return self.loss

# Sane check
if len(sys.argv) <= 1:
    print("Usage mnist.py [MNIST.pckl]")
    sys.exit(1)

# Loading data
x_all, y_all = load_data(sys.argv[1])
x_train, x_test = np.split(x_all, [60000])
y_train, y_test = np.split(y_all, [60000])
train_size = len(x_train)
test_size  = len(x_test)

# Init model
model = L.Classifier(MLP())
optimizer = optimizers.SGD()
optimizer.setup(model)

# Training begins here
UF.trace("Begin training")
for epoch in range(EPOCH):
    UF.trace("Epoch %d" % (epoch+1))
    indexes = np.random.permutation(train_size)
    for i in range(0, train_size, BATCH):
        x = Variable(x_train[indexes[i:i+BATCH]])
        t = Variable(y_train[indexes[i:i+BATCH]])
        model.zerograds()
        loss = model(x,t)
        loss.backward()
        optimizer.update()

# Testing begins here
UF.trace("Begin Testing")
sum_loss, sum_accuracy = 0, 0
for i in range(0, test_size, BATCH):
    x = Variable(x_test[i: i+BATCH])
    t = Variable(y_test[i: i+BATCH])
    loss = model(x,t)
    sum_loss += loss.data * batchsize
    sum_accuracy = model.accuracy.data * BATCH

# Calculating final result
mean_loss = sum_loss/ test_size
mean_accuracy = sum_accuracy / test_size

UF.trace("Accuracy:", mean_accuracy)

