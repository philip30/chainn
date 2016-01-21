import math

def check_nan(arr):
    for x in arr.data:
        for i in x:
            if math.isnan(float(i)):
                raise ValueError()
