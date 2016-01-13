#!/bin/bash

set -e
set -o xtrace

test_dir=/tmp/chainn-test/lm
inp=$1
dev=$2
tst=$3
train=$4
test=$5

mkdir -p $test_dir
python3 $train --hidden 5 --embed 5 --model_out $test_dir/tmp.model --epoch 2 --use_cpu --dev $dev < $inp
python3 $test --init_model $test_dir/tmp.model < $tst
