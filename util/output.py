class DecodingOutput:
    
    def __init__(self, decoding=None, alignment=None):
        self.__data = {}
        self.__data["decode"] = decoding
        self.__data["alignment"] = alignment

    def __getitem__(self, key):
        return self.__data[key]
