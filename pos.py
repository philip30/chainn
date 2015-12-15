#!/usr/bin/env python3 

import sys
import argparse
import numpy as np

from collections import defaultdict
from chainn import functions as UF
from chainn.model import RNNParallelSequence

def parse_args():
    parser = argparse.ArgumentParser("Program for multi-class classification using multi layered perceptron")
    parser.add_argument("--batch", type=int, help="Minibatch size", default=64)
    parser.add_argument("--init_model", required=True, type=str, help="Initiate the model from previous")
    parser.add_argument("--model", type=str, choices=["lstm", "rnn"], default="lstm")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Variable
    batch_size = args.batch

    # Setup model
    UF.trace("Setting up classifier")
    model = RNNParallelSequence(args, use_gpu=not args.use_cpu)
    X, Y  = model.get_vocabularies()

    # data
    UF.trace("Loading test data + dictionary from stdin")
    test, sent_ids = load_data(sys.stdin, X, args.batch)
       
    # POS Tagging
    output_collector = {}
    UF.trace("Start Tagging")
    for batch, batch_id in zip(test, sent_ids):
        tag_result = model.predict(batch)
        for inp, result, id in zip(batch, tag_result, batch_id):
            output_collector[id] = Y.str_rpr(result)
            
            if args.verbose:
                inp    = [X.tok_rpr(x) for x in inp]
                result = [Y.tok_rpr(x) for x in result]
                print(" ".join(str(x) + "_" + str(y) for x, y in zip(inp, result)), file=sys.stderr)

    # Printing all output
    for _, result in sorted(output_collector.items(), key=lambda x:x[0]):
        print(result)

def load_data(fp, x_ids, batch_size):
    holder        = defaultdict(lambda:[])
    # Reading in the data
    for sent_id, line in enumerate(fp):
        sent          = line.strip().lower().split()
        words, labels = [], []
        for word in sent:
            words.append(x_ids[word])
        holder[len(words)].append((sent_id,words))

    # Convert to appropriate data structure
    X = []
    sent_ids = []
    for src_len, items in sorted(holder.items(), key=lambda x:x[0]):
        item_count = 0
        x_batch    = []
        sent_batch = []
        for sent_id, words in items:
            x_batch.append(words)
            sent_batch.append(sent_id)
            item_count += 1

            if item_count % batch_size == 0:
                X.append(x_batch)
                sent_ids.append(sent_batch)
                x_batch, sent_batch = [], []
        if len(x_batch) != 0:
            X.append(x_batch)
            sent_ids.append(sent_batch)
    return X, sent_ids

if __name__ == "__main__":
    main()

