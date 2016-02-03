
from collections import defaultdict
from chainn.util import Vocabulary

from chainn.util import functions as UF

def strip_split(line):
    return line.strip().split()

def unsorted_batch(x_batch, SRC):
    max_len = max(len(x) for x in x_batch)
    for i in range(len(x_batch)):
        x_batch[i] += [SRC.eos_id() for _ in range(max_len-len(x_batch[i]))]
    return x_batch

def load_train_data(data, SRC, TRG, batch_size=1, src_count=None, trg_count=None, x_cut=1, y_cut=1, replace_unknown=False, debug=False):
    rep_rare = lambda vocab, w, count, cut: vocab[w] if count is None or count[w] > cut else vocab.unk_id()
    rep_unk  = lambda vocab, w: vocab[w] if w in vocab else vocab.unk_id()
    convert_to_id = lambda vocab, w, count, cut: rep_unk(vocab, w) if replace_unknown else rep_rare(vocab, w, count, cut)

    item_count = 0
    ret = []
    x_batch, y_batch = [], []
    holder = []
    for src, trg in data:
        src = [convert_to_id(SRC, word, src_count, x_cut) for word in src]
        trg = [convert_to_id(TRG, word, trg_count, y_cut) for word in trg]
        holder.append((src, trg))

    for src, trg in sorted(holder, key=lambda x: len(x[0]), reverse=debug):
        x_batch.append(src), y_batch.append(trg)
        item_count += 1

        if item_count % batch_size == 0:
            ret.append((unsorted_batch(x_batch, SRC), unsorted_batch(y_batch, TRG)))
            x_batch, y_batch = [], []
    if len(x_batch) != 0:
        ret.append((unsorted_batch(x_batch, SRC), unsorted_batch(y_batch, TRG)))
    UF.trace("SRC size:", len(SRC))
    UF.trace("TRG size:", len(TRG))
    return ret

def load_test_data(lines, SRC, batch_size=1, preprocessing=strip_split):
    rep_rare = lambda vocab, w: vocab[w] if w in vocab else vocab.unk_id()
    
    item_count = 0
    x_batch    = []
    for src in lines:
        src = [rep_rare(SRC, word) for word in preprocessing(src)]
        x_batch.append(src)
        item_count += 1

        if item_count % batch_size == 0:
            yield unsorted_batch(x_batch, SRC)
            x_batch = []
    
    if len(x_batch) != 0:
        yield unsorted_batch(x_batch, SRC)

"""
* POS TAGGER *
"""
def load_pos_train_data(lines, batch_size=1, cut_threshold=1):
    SRC, TRG = Vocabulary(unk=True, eos=True), Vocabulary(unk=False, eos=True)
    data     = []
    w_count  = defaultdict(lambda: 0)

    # Reading in the data
    for line in lines:
        sent          = line.strip().split()
        words, labels = [], []
        for word in sent:
            word, tag = word.split("_")
            words.append(word)
            labels.append(tag)
            w_count[word] += 1
        data.append((words,labels))

    # Data generator
    data_generator = load_train_data(data, SRC, TRG, batch_size, src_count=w_count, x_cut=cut_threshold)
    
    # Return
    return SRC, TRG, data_generator

def load_pos_test_data(lines, SRC, batch_size=1):
    return load_test_data(lines, SRC, batch_size)

"""
* NMT *
"""
def load_nmt_train_data(src, trg, batch_size=1, cut_threshold=1, debug=False):
    src_count = defaultdict(lambda:0)
    trg_count = defaultdict(lambda:0)
    SRC  = Vocabulary(unk=True, eos=True)
    TRG  = Vocabulary(unk=True, eos=True)
    data = []
    # Reading in data
    for sent_id, (src_line, trg_line) in enumerate(zip(src, trg)):
        src_line = src_line.strip().lower().split() + [SRC.eos()]
        trg_line = trg_line.strip().lower().split() + [TRG.eos()]

        for word in src_line:
            src_count[word] += 1
        for word in trg_line:
            trg_count[word] += 1

        data.append((src_line, trg_line))
   
    # Data generator
    data_generator = load_train_data(data, SRC, TRG, batch_size, \
            src_count=src_count, trg_count=trg_count, x_cut=cut_threshold, y_cut=cut_threshold, debug=debug)
    
    # Return
    return SRC, TRG, data_generator
    
def load_nmt_test_data(src, SRC, batch_size=1):
    def preprocessing(line):
        return line.strip().split() + [SRC.eos()]
    
    return load_test_data(src, SRC, batch_size, preprocessing)

"""
* LANGUAGE MODEL *
"""
def load_lm_data(lines, SRC=None, batch_size=1, cut_threshold=1):
    replace_unk = SRC is not None
    if SRC is None:
        SRC = Vocabulary()
        SRC["<s>"], SRC["</s>"]

    count  = defaultdict(lambda:0)
    data   = []
    # Reading and counting the data
    for sent_id, line in enumerate(lines):
        sent = ["<s>"] + line.strip().lower().split() + ["</s>"]
        words, next_w = [], []
        for i, tok in enumerate(sent):
            count[tok] += 1
            if i < len(sent)-1:
                words.append(sent[i])
                next_w.append(sent[i+1])
        data.append((words, next_w))

    # Data generator
    data_generator = load_train_data(data, SRC, SRC, batch_size, \
            src_count=count, trg_count=count, x_cut=cut_threshold, y_cut=cut_threshold,\
            replace_unknown=replace_unk)

    return SRC, data_generator

