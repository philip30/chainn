
class DecodingOutput:
    def __init__(self, output_dict):
        self.__attr = []
        for key, value in output_dict.items():
            setattr(self, key, value)
            

