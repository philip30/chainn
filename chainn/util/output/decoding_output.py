
class DecodingOutput:
    def __init__(self, output_dict):
        self.__attr = []
        for key, value in output_dict.items():
            setattr(self, key, value)
            
    def __str__(self):
        ret = []
        for key in self.__attr:
            value = getattr(self, key)
            ret.append(str(key)+":"+str(value))
        return ",".join(ret)

