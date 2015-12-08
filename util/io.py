from .functions import vtos, stov

# Copied and modified from https://github.com/odashi/chainer_examples/blob/master/util/model_file.py
class ModelFile:
    def __init__(self, fp):
        self.__fp = fp

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__fp.close()
        return False

    def write(self, x):
        print(x, file=self.__fp)

    def __write_vector(self, x):
        self.write(vtos(x))

    def __write_matrix(self, x):
        for row in x:
            self.__write_vector(row)
    
    def read(self):
        return next(self.__fp).strip()

    def __read_vector(self, x, tp):
        data = stov(self.read(), tp)
        for i in range(len(data)):
            x[i] = data[i]

    def __read_matrix(self, x, tp):
        for row in x:
            self.__read_vector(row, tp)

    def write_embed(self, f):
        self.__write_matrix(f.W.data)

    def write_linear(self, f):
        self.__write_matrix(f.W.data)
        self.__write_vector(f.b.data)

    def read_embed(self, f):
        self.__read_matrix(f.W.data, float)

    def read_linear(self, f):
        self.__read_matrix(f.W.data, float)
        self.__read_vector(f.b.data, float)

    def get_file_pointer(self):
        return self.__fp

