import numpy as np
import time
import sys
from classifier import *

times = int(sys.argv[1])
dtype = sys.argv[2]
nlayers = int(sys.argv[3])
size = int(sys.argv[4])
epsilon = sys.argv[5]
activation_name = sys.argv[6]


props = GlobalProperties(dtype, epsilon)
stabilizer = Stabilizer(props)
functions = Functions(props, stabilizer)
scale = props.f_1 / np.log(props.f_1 + props.dtype(size), dtype=props.dtype)

if activation_name == "sigmoid" :
    activation = functions.sigmoid
else :
    activation = functions.tanh

def build_layers(nlayers, size, dtype):
    layers, biases = [],[]
    for l in range(nlayers) :
        weights = np.random.rand(size, size)
        weights = (np.array(weights, dtype=props.dtype) - props.dtype(0.5)) * scale
        layers.append(weights)
        bias = (np.array(np.random.rand(size, 1), dtype=props.dtype) - props.dtype(0.5) * scale)
        biases.append(bias)
    return layers, biases

def forward(layers,biases) :
    inp = np.array(np.random.rand(size, 1), dtype = props.dtype)
    out = np.array(np.random.rand(size, 1), dtype = props.dtype)
    for layer,bias in zip(layers,biases) :
        np.matmul(layer, inp, out=out)
        out += bias
        out = activation(out) * scale
        temp = inp
        inp = out
        out = temp

def benchmark() :
    start = time.time_ns()
    layers, biases = build_layers(nlayers, size, dtype)
    for i in range(times) :
        forward(layers, biases)
    end = time.time_ns()
    return (end - start) / 1000000.0


print(benchmark())
