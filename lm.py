#!/usr/bin/env python3 

import sys
import argparse
import numpy as np
import math

from collections import defaultdict
from chainn import functions as UF
from chainn.model import RNNParallelSequence

def parse_args():
    parser = argparse.ArgumentParser("Program for multi-class classification using multi layered perceptron")
    parser.add_argument("--batch", type=int, help="Minibatch size", default=1)
    parser.add_argument("--init_model", required=True, type=str, help="Initiate the model from previous")
    parser.add_argument("--model", type=str, choices=["lstm", "rnn"], default="lstm")
    parser.add_argument("--operation", choices=["sppl", "cppl"], help="sppl: Sentence-wise ppl\ncppl: Corpus-wise ppl", default="sppl")
    parser.add_argument("--gen", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Variable
    batch_size = args.batch

    # Setup model
    UF.trace("Setting up classifier")
    model = RNNParallelSequence(args, use_gpu=not args.use_cpu, collect_output=True)
    X, _  = model.get_vocabularies()

    # data
    UF.trace("Loading test data + dictionary from stdin")
    word, next_word, sent_ids = load_data(sys.stdin, args.batch, X)
       
    # POS Tagging
    output_collector = {}
    UF.trace("Start Calculating PPL")
    for x_data, y_data, batch_id in zip(word, next_word, sent_ids):
        accum_loss, _, output = model.train(x_data, y_data, update=False)
        
        accum_loss = accum_loss / len(x_data)
        for inp, result, id in zip(x_data, output, batch_id):
            output_collector[id] = (X.str_rpr(result), accum_loss)
            
            if args.verbose:
                inp    = [X.tok_rpr(x) for x in inp]
                result = [X.tok_rpr(x) for x in result]
                print("INP:", " ".join(inp), file=sys.stderr)
                print("OUT:", " ".join(result), file=sys.stderr)
                print("PPL:", math.exp(accum_loss), file=sys.stderr)

    # Printing all output
    gen_fp = open(args.gen, "w") if args.gen else None
    operation = args.operation
    if operation == "sppl":
        for _, (result, accum_loss) in sorted(output_collector.items(), key=lambda x:x[0]):
            print(math.exp(accum_loss))
            if gen_fp is not None:
                print(result, file=gen_fp)
    elif operation == "cppl":
        total_loss = 0
        for _, (result, accum_loss) in output_collector.items():
            total_loss += accum_loss
        total_loss = float(total_loss) / len(output_collector)
        print(math.exp(total_loss))
    if gen_fp is not None:
        gen_fp.close()


def load_data(fp, batch_size, x_ids):
    holder        = defaultdict(lambda:[])
    # Reading in the data
    for sent_id, line in enumerate(fp):
        sent          = ["<s>"] + line.strip().lower().split() + ["</s>"]
        words, next_w = [], []
        for i in range(len(sent)-1):
            words.append(x_ids[sent[i]] if sent[i] in x_ids else x_ids[x_ids.unk()])
            next_w.append(x_ids[sent[i+1]] if sent[i+1] in x_ids else x_ids[x_ids.unk()])
        holder[len(words)].append((sent_id, words, next_w))

    # Convert to appropriate data structure
    X, Y, sent_ids = [], [], []
    for src_len, items in sorted(holder.items(), key=lambda x:x[0]):
        item_count = 0
        x_batch, y_batch = [], []
        sent_batch = []
        for sent_id, words, next_words in items:
            x_batch.append(words)
            y_batch.append(next_words)
            sent_batch.append(sent_id)
            item_count += 1

            if item_count % batch_size == 0:
                X.append(x_batch)
                Y.append(y_batch)
                sent_ids.append(sent_batch)
                x_batch, y_batch = [], []
        if len(x_batch) != 0:
            X.append(x_batch)
            Y.append(y_batch)
            sent_ids.append(sent_batch)
    return X, Y, sent_ids

if __name__ == "__main__":
    main()

