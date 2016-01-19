
import sys
import unicodedata

def count_len(data):
    ret = 0
    for c in data:
        if unicodedata.east_asian_width(c) == 'W':
            ret += 2
        else:
            ret += 1
    return ret

class AlignmentVisualizer:
    @staticmethod
    def print(data, start_index, src, trg, src_voc, trg_voc, fp=sys.stderr, precision=3, margin=2):
        # Printing for every input
        for index, out in enumerate(data):
            print(index+start_index, file=fp)
            str_data = [["" for _ in range(len(out)+1)] for _ in range(len(out[0])+1)]
            src_data = src[index]
            trg_data = trg[index]
            eos = trg_voc.eos_id()

            if eos in trg_data:
                trg_data = trg_data[:trg_data.index(eos)+1]

            # Header
            for i in range(len(src_data)):
                str_data[i+1][0] = src_voc.tok_rpr(src_data[i])
            for i in range(len(trg_data)):
                str_data[0][i+1] = trg_voc.tok_rpr(trg_data[i])
            
            # Filling
            format = "%." + str(precision) + "f"
            for i in range(len(out)):
                for j in range(len(out[i])):
                    str_data[j+1][i+1] = format % (out[i][j])

            # Max_len
            max_len = [0 for _ in range(len(str_data[0]))]
            for i in range(len(str_data)):
                for j in range(len(str_data[i])):
                    len_str = count_len(str_data[i][j])
                    max_len[j] = max(len_str, max_len[j])

            # Generate space
            for i in range(len(str_data)):
                for j in range(len(str_data[i])):
                    now_len = count_len(str_data[i][j])
                    remainder = max_len[j] - now_len + margin
                    front = remainder // 2
                    back = remainder - front
                    str_data[i][j] = (" " * front) + str_data[i][j] + (" " * back)
            
            # Printing 
            for i in range(len(str_data)):
                print(" ".join(str_data[i]), file=fp)
        return
    

