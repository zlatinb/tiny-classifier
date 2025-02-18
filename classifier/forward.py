#!/usr/bin/python3
import numpy as np
from . import *

class ForwardPass :

    def __init__(self, props) :
        self.props = props
        stabilizer = Stabilizer(props)
        self.functions = Functions(props, stabilizer)
        self.scale = props.f_1 / np.log(props.f_1 + props.dtype(self.props.size), dtype=props.dtype)
        self.activation = self.functions.default_activation()


    def forward(self, layers,biases) :
        inp = np.array(np.random.rand(self.props.size, 1), dtype = self.props.dtype)
        out = np.array(np.random.rand(self.props.size, 1), dtype = self.props.dtype)
        for layer,bias in zip(layers,biases) :
            np.matmul(layer, inp, out=out)
            out += bias
            out = self.activation(out) * self.scale
            temp = inp
            inp = out
            out = temp

