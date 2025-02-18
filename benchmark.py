#!/usr/bin/python3
import argparse as ap
from bench import Driver

if __name__  == "__main__" :
    
    parser = ap.ArgumentParser(
            description="Benchmarks the classifier on your CPU")
    parser.add_argument('--nepochs', '-n', type=int, help='How many epochs to run', required = True)
    parser.add_argument('--precision', '-p', type=str, choices = ["f32","f64"], 
        help="What precision to use.  Possible values are f32 and 64",
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
    args.name = None

    driver = Driver(args)
    total = driver.run_bench()

    if args.name is not None :
        print(f"Completed \"{args.name}\" in {total} ms")
    else :
        print(f"Total time for {args.nepochs} run(s) was {total} ms")
