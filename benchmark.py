#!/usr/bin/python3
from numpy import float64
import argparse as ap
import yaml
from bench import Driver

YAML_VERSION = 1
ACTIVATIONS = ("sigmoid", "tanh")
PRECISIONS = ("f4","f8")

class Args :
    def __init__(self) :
        pass

if __name__  == "__main__" :
    parser = ap.ArgumentParser(description="Benchmarks the classifier on your CPU") 
    parser.add_argument("--config", "-c", type=str, help="Configuration file in YAML", required = True)
    args = parser.parse_args()
   
    with open(args.config) as cfg :
        parsed = yaml.safe_load(cfg)
        if not parsed :
            raise ValueError(f"Couldn't load file {args.config}")
        if parsed['version'] > YAML_VERSION :
            raise ValueError(f"Maximum config version supported is {YAML_VERSION}")

        def least(where, name, threshold, dt = int, required = True) :
            val = where.get(name, None)
            if not required and not val :
                return None
            rv = dt(val)
            if rv < threshold :
                raise ValueError(f"{name} must be at least {value}")
            return rv
        def is_in(where, name, domain, required = True) :
            if not required and name in where :
                return None
            if where[name] not in set(domain) :
                raise ValueError(f"Allowed values for {name} are {domain}")
            return where[name]

        benchmark = parsed['benchmark']
        args = Args()
        args.name = benchmark.get("name", None)
        args.nepochs = least(benchmark, "nepochs", 1)
        args.layers = least(benchmark['shape'], "layers", 1)
        epsilons = benchmark['math']['epsilons']
        args.epsilon_exp = least(epsilons, "exp", 0, dt=float64)
        args.epsilon_square = least(epsilons, "square", 0, dt=float64)
        args.dimension = least(benchmark['shape'], "dimension", 1)
        args.precision = is_in(benchmark['math'], "precision", PRECISIONS)
        args.activation = is_in(benchmark['math'], "activation", ACTIVATIONS)
        args.threads = None
        if "performance" in benchmark and benchmark['performance'] :
            args.threads = least(benchmark['performance'], "threads", 1, required = False)

        driver = Driver(args)
        total = driver.run_bench()

        if args.name is not None :
            print(f"Completed \"{args.name}\" in {total} ms")
        else :
            print(f"Total time for {args.nepochs} run(s) was {total} ms")
