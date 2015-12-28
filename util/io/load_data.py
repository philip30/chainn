
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


