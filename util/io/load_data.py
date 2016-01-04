
from collections import defaultdict
from chainn.util import Vocabulary

def load_pos_train_data(lines, batch_size=1, cut_threshold=1):
    x_ids, y_ids  = Vocabulary(), Vocabulary(unk=False)
    holder        = defaultdict(lambda:[])
    stat          = defaultdict(lambda: 0)

    # Reading in the data
    for line in lines:
        sent          = line.strip().split()
        words, labels = [], []
        for word in sent:
            word, tag = word.split("_")
            words.append(word)
            labels.append(y_ids[tag])
            stat[word] += 1
        holder[len(words)].append((words,labels))

    # Convert to appropriate data structure
    X, Y = [], []
    for src_len, items in sorted(holder.items(), key=lambda x:x[0]):
        item_count = 0
        x_batch, y_batch = [], []
        for words, labels in items:
            words = list(map(lambda x: x_ids.unk_id() if stat[x] <= cut_threshold else x_ids[x], words))
            x_batch.append(words)
            y_batch.append(labels)
            item_count += 1

            if item_count % batch_size == 0:
                X.append(x_batch)
                Y.append(y_batch)
                x_batch, y_batch = [], []
        if len(x_batch) != 0:
            X.append(x_batch)
            Y.append(y_batch)
    return X, Y, x_ids, y_ids

def load_pos_test_data(lines, x_ids, batch_size=1):
    holder        = defaultdict(lambda:[])
    # Reading in the data
    for sent_id, line in enumerate(lines):
        sent          = line.strip().split()
        words, labels = [], []
        for word in sent:
            if word in x_ids:
                words.append(x_ids[word])
            else:
                words.append(x_ids.unk_id())
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

def load_lm_data(lines, x_ids=None, batch_size=1, cut_threshold=1):
    replace_unk = x_ids is not None
    if x_ids is None:
        x_ids = Vocabulary()
        x_ids["<s>"], x_ids["</s>"]

    count  = defaultdict(lambda:0)
    holder = defaultdict(lambda:[])
    # Reading and counting the data
    for sent_id, line in enumerate(lines):
        sent = ["<s>"] + line.strip().lower().split() + ["</s>"]
        words, next_w = [], []
        for i, tok in enumerate(sent):
            count[tok] += 1
            if i < len(sent)-1:
                words.append(sent[i])
                next_w.append(sent[i+1])
        holder[len(words)].append([sent_id, words, next_w])

    id_train = lambda x: x_ids[x] if count[x] > cut_threshold else x_ids.unk_id()
    id_rep = lambda x: x_ids[x] if  x in x_ids else x_ids.unk_id()
    convert_to_id = id_rep if replace_unk else id_train
    # Convert to appropriate data structure
    X, Y, ids = [], [], []
    for src_len, items in sorted(holder.items(), key=lambda x:x[0]):
        item_count = 0
        x_batch, y_batch, id_batch = [], [], []
        for sent_id, words, next_words in items:
            word = list(map(convert_to_id, words))
            nw   = list(map(convert_to_id, next_words))
            x_batch.append(word)
            y_batch.append(nw)
            id_batch.append(sent_id)
            item_count += 1

            if item_count % batch_size == 0:
                X.append(x_batch)
                Y.append(y_batch)
                ids.append(id_batch)
                x_batch, y_batch, id_batch = [], [], []
        if len(x_batch) != 0:
            X.append(x_batch)
            Y.append(y_batch)
            ids.append(id_batch)
    return X, Y, x_ids, ids

def load_nmt_train_data(src, trg, batch_size=1, cut_threshold=1):
    data = defaultdict(lambda:[])
    SRC  = Vocabulary(unk=True, eos=True)
    TRG  = Vocabulary(unk=True, eos=True)

    src_count = defaultdict(lambda:0)
    trg_count = defaultdict(lambda:0)

    # Reading in data
    for sent_id, (src_line, trg_line) in enumerate(zip(src, trg)):
        src_line = src_line.strip().lower().split() + [SRC.eos()]
        trg_line = trg_line.strip().lower().split() + [TRG.eos()]

        for word in src_line:
            src_count[word] += 1
        for word in trg_line:
            trg_count[word] += 1
        
        data[len(src_line), len(trg_line)].append((sent_id, src_line, trg_line))
        
    # Convert to id
    rep_rare = lambda x, y, z: x[y] if z[y] > cut_threshold else x.unk_id()
    x_data, y_data = [], []
    for key_len, items in sorted(data.items(), key=lambda x: x[0]):
        item_count = 0

        x_batch, y_batch = [], []
        for sent_id, src_line, trg_line in items:
            src_line = [rep_rare(SRC, word, src_count) for word in src_line]
            trg_line = [rep_rare(TRG, word, trg_count) for word in trg_line]
            x_batch.append(src_line)
            y_batch.append(trg_line)
            item_count += 1
            
            if item_count % batch_size == 0:
                x_data.append(x_batch)
                y_data.append(y_batch)
                x_batch, y_batch = [], []
        if len(x_batch) != 0:
            x_data.append(x_batch)
            y_data.append(y_batch)
    return x_data, y_data, SRC, TRG

def load_nmt_test_data(src, SRC, batch_size=1):
    data = defaultdict(lambda:[])

    # Reading in data
    for sent_id, src_line in enumerate(src):
        src_line = src_line.strip().lower().split() + [SRC.eos()]

        data[len(src_line)].append((sent_id, src_line))
        
    # Convert to id
    rep_rare = lambda x, y: x[y] if y in x else x.unk_id()
    x_data, ids = [], []
    for key_len, items in sorted(data.items(), key=lambda x: x[0]):
        item_count = 0

        x_batch, id_batch = [], []
        for sent_id, src_line in items:
            src_line = [rep_rare(SRC, word) for word in src_line]
            x_batch.append(src_line)
            id_batch.append(sent_id)
            item_count += 1
            
            if item_count % batch_size == 0:
                x_data.append(x_batch)
                ids.append(id_batch)
                x_batch, id_batch = [], []
        if len(x_batch) != 0:
            x_data.append(x_batch)
            ids.append(id_batch)
    return x_data, ids

