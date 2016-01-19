import sys
import unicodedata

class DecodingOutput:
    def __init__(self, y=None, a=None):
        self.y = y
        self.a = a

class AlignmentVisualizer:
    def __init__(self, out_dir):
        self.__fp = None
        if out_dir:
            self.__fp = open(out_dir, "w")
            counter = 0

    def close(self):
        if self.__fp is not None:
            self.__fp.close()

    def print(self, data, src, trg, src_voc, trg_voc, precision=3, margin=2):
        if self.__fp is None:
            return

        # Printing for every input
        for index, out in enumerate(data):
            str_data = [["" for _ in range(len(out)+1)] for _ in range(len(out[0])+1)]
            src_data = src[index]
            trg_data = trg[index]
            
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
                    len_str = self.__count_len(str_data[i][j])
                    max_len[j] = max(len_str, max_len[j])

            # Generate space
            for i in range(len(str_data)):
                for j in range(len(str_data[i])):
                    now_len = self.__count_len(str_data[i][j])
                    remainder = max_len[j] - now_len + margin
                    front = remainder // 2
                    back = remainder - front
                    str_data[i][j] = (" " * front) + str_data[i][j] + (" " * back)
            
            # Printing 
            for i in range(len(str_data)):
                print(" ".join(str_data[i]), file=self.__fp)
        return
    
    def __count_len(self, data):
        ret = 0
        for c in data:
            if unicodedata.east_asian_width(c) == 'W':
                ret += 2
            else:
                ret += 1
        return ret

