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

[ -d $test_dir/tmp.model ] && rm -rf $test_dir/tmp.model

for (( epoch=0; epoch < 10; epoch++ )); do
    if [ $epoch != 0 ]; then
        init_model="--init_model $test_dir/tmp.model"
    fi
    python3 $train --hidden 5 --embed 5 --batch 3 --epoch 10 $init_model --one_epoch --src $src --trg $trg --model $model --unk_cut 0 --depth 1 --seed 20 --model_out $test_dir/tmp.model > /dev/null
done

