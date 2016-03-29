#!/bin/bash

set -e
set -o xtrace

test_dir=/tmp/chainn-test/pos
inp=$1
tst=$2
train=$3
test=$4

mkdir -p $test_dir
for model in lstm; do 
    python3 $train --hidden 5 --embed 5 --model_out $test_dir/tmp.model --epoch 2 --use_cpu --model $model < $inp
    python3 $test --init_model $test_dir/tmp.model < $tst
done
