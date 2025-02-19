# Tiny Classifier - a configurable neural net classifier

This project aims to create a small neural network capable of categorizing input.  It uses in pure Python and runs on CPUs.  

I started it as a learning project to help me in my studies of ML math, but I realized the final output could be useful for specialized inference on a budget.  To this end, I am very mindful of performance - if you look at the code you will see heavy emphasis on benchmarking.

### Installation


Install the `threadpoolctl` package and launch the `benchmark.py` file to run a quick benchmark:
```bash
pip install threadpoolctl
./benchmark.py --help
usage: benchmark.py [-h] --nepochs NEPOCHS --precision PRECISION --layers LAYERS --dimension DIMENSION --epsilon
                    EPSILON --activation {sigmoid,tanh}

Benchmarks the classifier on your CPU

options:
  -h, --help            show this help message and exit
  --nepochs NEPOCHS, -n NEPOCHS
                        How many epochs to run
  --precision PRECISION, -p PRECISION
                        What precision to use. Possible values are f32 and 64
  --layers LAYERS, -l LAYERS
                        How many hidden layers to have in the neural net
  --dimension DIMENSION, -d DIMENSION
                        How many perceptrons per hidden layer
  --activation {sigmoid,tanh}, -a {sigmoid,tanh}
                        Type of activation function to use. Available are sigmoid and tanh
  --threads THREADS, -t THREADS
                        Maximum number of threads to use. Optional, defaults to max available.
```
Here are the results from my laptop:
```
./benchmark.py -p f32 -n 50 -l 100 -d 1024 -a sigmoid
Total time for 50 run(s) was 3327.771058 ms
```

### Benchmarking
The above tool works for quick tests; there is also a YAML-driven tool that supports fine-grained and elaborate benchmarking.  In-depth discussion is in [BENCHMARKING.md]

[BENCHMARKING.md]: https://github.com/zlatinb/tiny-classifier/blob/main/BENCHMARKING.md

