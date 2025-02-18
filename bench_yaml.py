#!/usr/bin/python3
import argparse as ap
import yaml
from bench import Driver

YAML_VERSION = 1
ACTIVATIONS = ("sigmoid", "tanh")
PRECISIONS = ("f32","f64")

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

        def least(where, name, value) :
            if where[name] < value :
                raise ValueError(f"{name} must be at least {value}")
            return where[name]
        def is_in(where, name, domain) :
            if where[name] not in set(domain) :
                raise ValueError(f"Allowed values for {name} are {domain}")
            return where[name]

        benchmark = parsed['benchmark']
        args = Args()
        args.name = benchmark.get("name", None)
        args.nepochs = least(benchmark, "nepochs", 1)
        args.layers = least(benchmark['shape'], "layers", 1)
        args.dimension = least(benchmark['shape'], "dimension", 1)
        args.epsilon = least(benchmark['math'], "epsilon", 0)
        args.precision = is_in(benchmark['math'], "precision", PRECISIONS)
        args.activation = is_in(benchmark['math'], "activation", ACTIVATIONS)
        args.threads = least(benchmark['performance'], "threads", 1)

        driver = Driver(args)
        total = driver.run_bench()

        if args.name is not None :
            print(f"Completed \"{args.name}\" in {total} ms")
        else :
            print(f"Total time for {args.nepochs} run(s) was {total} ms")
