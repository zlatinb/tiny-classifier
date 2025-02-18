import numpy as np
import time
import threadpoolctl as tpc
from classifier import *

class Driver :
    def __init__(self, args) :
        self.args = args
        self.props = GlobalProperties(args)
        props = self.props
        stabilizer = Stabilizer(props)
        scale = props.f_1 / np.log(props.f_1 + props.dtype(args.dimension), dtype=props.dtype)

        layers, biases = [],[]
        for l in range(args.layers) :
            weights = np.random.rand(args.dimension, args.dimension)
            weights = (np.array(weights, dtype=props.dtype) - props.dtype(0.5)) * scale
            layers.append(weights)
            bias = (np.array(np.random.rand(args.dimension, 1), dtype=props.dtype) - props.dtype(0.5) * scale)
            biases.append(bias)

        self.layers = layers
        self.biases = biases

    def run_bench(self) :
        with tpc.threadpool_limits(limits=self.args.threads, user_api='blas'):
            forward = ForwardPass(self.props)
            start = time.time_ns()
            for i in range(self.args.nepochs) :
                forward.forward(self.layers,self.biases)
            end = time.time_ns()
            total = ( end - start ) / 1000000.0
        return total
