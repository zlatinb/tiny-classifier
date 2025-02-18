#!/usr/bin/python3
import threadpoolctl as tpc
import argparse as ap
import numpy as np
import time, sys
from classifier import *


if __name__  == "__main__" :
    
    parser = ap.ArgumentParser(
            description="Benchmarks the classifier on your CPU")
    parser.add_argument('--nepochs', '-n', type=int, help='How many epochs to run', required = True)
    parser.add_argument('--precision', '-p', type=str, choices = ["f32","f64"], help="What precision to use.  Possible values are f32 and 64",
        required = True)
    parser.add_argument('--layers', '-l', type=int, help="How many hidden layers to have in the neural net",
        required = True)
    parser.add_argument('--dimension', '-d', type=int, help="How many perceptrons per hidden layer",
        required = True)
    parser.add_argument('--epsilon', '-e', type=float, help="Value of epsilon to use for numerical stability",
        required = True)
    parser.add_argument('--activation', "-a", type=str, choices=["sigmoid","tanh"], help="Type of activation function to use.  Available are sigmoid and tanh", required = True)
    parser.add_argument("--threads", "-t", type=int, 
        help="Maximum number of threads to use.  Optional, defaults to max available.")
    args = parser.parse_args()

    props = GlobalProperties(args)
    stabilizer = Stabilizer(props)
    functions = Functions(props, stabilizer)
    scale = props.f_1 / np.log(props.f_1 + props.dtype(args.dimension), dtype=props.dtype)

    layers, biases = [],[]
    for l in range(args.layers) :
        weights = np.random.rand(args.dimension, args.dimension)
        weights = (np.array(weights, dtype=props.dtype) - props.dtype(0.5)) * scale
        layers.append(weights)
        bias = (np.array(np.random.rand(args.dimension, 1), dtype=props.dtype) - props.dtype(0.5) * scale)
        biases.append(bias)
 
    with tpc.threadpool_limits(limits=args.threads, user_api='blas'):
        forward = ForwardPass(props)
        start = time.time_ns()
        for i in range(args.nepochs) :
            forward.forward(layers,biases)
        end = time.time_ns()
        total = ( end - start ) / 1000000.0

    print(f"Total time for {args.nepochs} run(s) was {total} ms")
