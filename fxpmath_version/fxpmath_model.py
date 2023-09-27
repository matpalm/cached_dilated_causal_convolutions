
import pickle
from fxpmath import Fxp




class FxpModel(object):

    def __init__(self, weights_file,
                 n_word=16, n_int=4, n_frac=12):

        with open(weights_file, 'rb') as f:
            self.weights = pickle.load(f)

        self.fxp_util = FxpUtil(n_word, n_int, n_frac)

    def predict(self, x):
        pass