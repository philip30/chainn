import copy
from collections import defaultdict

def identity(x):
    return x

def parallel_batch(f, batchsize, pre_process=identity, post_process=identity, stuff=False):
    fp = list(map(lambda x: open(x, "r"), f)) # Open all files
    i = 0
    batch = []
    for lines in zip(*fp):
        batch.append(tuple(map(lambda x: pre_process(x),lines)))
        if (i+1) % batchsize == 0:
            yield post_process(batch)
            batch = []
        i += 1
    if len(batch) != 0:
        # Stuff the model to fit n*batchsize before yielding the last one
        if stuff:
            remainder = batchsize - len(batch)
            i = 0
            now_len = len(batch)
            while remainder != 0:
                batch.append(copy.deepcopy(batch[i % now_len]))
                i += 1
                remainder -= 1
        yield post_process(batch)
    list(map(lambda x: x.close(), fp)) # Close all files

def same_len_batch(f, batchsize, pre_process=identity, post_process=identity):
    fp = list(map(lambda x:open(x, "r"), f))
    data = defaultdict(lambda:[])
    for lines in zip(*fp):
        lines = tuple(map(lambda x: pre_process(x), lines))
        key = tuple(map(lambda x: len(x), lines))
        
        data[key].append(lines)
    list(map(lambda x: x.close(), fp))

    for count, sents in data.items():
        j = 0
        while j < len(sents):
            batch = sents[j:min(batchsize+j, len(sents))]
            yield post_process(batch)
            j += batchsize
    

