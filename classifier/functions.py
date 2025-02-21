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
        """to be used in backpropagation"""
        s = self.sigmoid(x)
        return s * (self.props.f_1 - s)

    def tanh(self,x) :
        np.tanh(x, out=x, dtype=self.props.dtype)
        return x

    def default_activation(self) :
        if self.props.activation == "sigmoid" :
            return self.sigmoid
        else :
            return self.tanh
