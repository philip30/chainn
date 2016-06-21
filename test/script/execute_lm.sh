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
python3 $train --hidden 5 --embed 5 --model_out $test_dir/tmp.model --epoch 2 --gpu 0 --dev $dev --seed 1991 < $inp

for op in  gen cppl sppl; do
    python3 $test --init_model $test_dir/tmp.model --operation $op < $tst
done

