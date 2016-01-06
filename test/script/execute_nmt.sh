#!/bin/bash

set -e
set -o xtrace

test_dir=/tmp/chainn-test/nmt
src=$1
trg=$2
tst=$3
train=$4
test=$5

mkdir -p $test_dir
for model in efattn encdec attn; do 
    python3 $train --debug --hidden 5 --embed 5 --model_out $test_dir/tmp.model --epoch 2 --use_cpu --model $model --src $src --trg $trg
    python3 $test --init_model $test_dir/tmp.model < $tst
done
