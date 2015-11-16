import copy

def identity(x):
    return x

def parallel_batch(src_file, trg_file, batchsize, pre_process=identity, post_process=identity):
    with open(src_file) as src_fp:
        with open(trg_file) as trg_fp:
            i = 0
            batch = []
            for src, trg in zip(src_fp, trg_fp):
                batch.append((pre_process(src), pre_process(trg)))

                if (i+1) % batchsize == 0:
                    yield post_process(batch)
                    batch = []
                i += 1
            if len(batch) != 0:
                remainder = batchsize - len(batch)
                i = 0
                now_len = len(batch)
                while remainder != 0:
                    batch.append(copy.deepcopy(batch[i % now_len]))
                    i += 1
                    remainder -= 1
                yield post_process(batch)


