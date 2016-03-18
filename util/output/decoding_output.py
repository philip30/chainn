
class DecodingOutput:
    def __init__(self, y=None, a=None):
        self.y = y
        self.a = a

    def __iter__(self):
        if self.a is not None:
            return iter(zip(self.y, self.a))
        else:
            return iter(self.y)
