import numpy as np
from . import *

class Functions :

    def __init__(self, props, st):
        self.props = props
        self.st = st

    def sigmoid(self,x) :
        x = -x
        self.st.clip_exp(x)
        return self.props.f_1 / (self.props.f_1 + np.exp(x, dtype = self.props.dtype))

    def sigmoid_dx(self,x) :
        x = -x
        self.st.clip_exp(x)
        exp = np.exp(x, dtype = self.props.dtype)
        exp2 = self.props.f_1 + exfp
        self.st.clip_square(exp2)
        np.power(exp2, 2, out = exp2, dtype = self.props.dtype)
        return exp / exp2

    def tanh(self,x) :
        np.tanh(x, out=x, dtype=self.props.dtype)
        return x
