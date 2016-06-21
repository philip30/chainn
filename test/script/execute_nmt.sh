#!/bin/bash

set -e
set -o xtrace

test_dir=/tmp/chainn-test/nmt
src=$1
trg=$2
tst=$3
train=$4
test=$5
model=$6
other=$7

mkdir -p $test_dir

python3 $train --gpu 0 --hidden 5 --embed 5 --model_out $test_dir/tmp.model --epoch 3 --model $model --src $src --trg $trg $other --unk_cut 0 --seed 20 --src_dev test/data/nmt.dev.en --trg_dev test/data/nmt.dev.ja > /dev/null
python3 $test --gpu 0 --init_model $test_dir/tmp.model --gen_limit 10 < $tst > /dev/null
