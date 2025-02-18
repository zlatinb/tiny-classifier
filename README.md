# Tiny Classifier - a configurable neural net classifier

This project aims to create a small neural network capable of categorizing input.  It is written in pure Python and runs on CPU.  Currently only the forward pass is implemented.  This tool may eventually be used to label large datasets after training on smaller labeled ones.

### Installation


Install the `threadpoolctl` package and launch the `benchmark.py` file:
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
  --epsilon EPSILON, -e EPSILON
                        Value of epsilon to use for numerical stability
  --activation {sigmoid,tanh}, -a {sigmoid,tanh}
                        Type of activation function to use. Available are sigmoid and tanh
  --threads THREADS, -t THREADS
                        Maximum number of threads to use. Optional, defaults to max available.
```
Here are the results from my laptop:
```bash
./benchmark.py -p f32 -n 20 -l 100 -d 50 -a sigmoid -e 0.0001
Total time for 20 run(s) was 22.169502 ms
```

### Benchmarking
The above tool works for quick tests; for more in-depth discussion take a look at [BENCHMARKING.md]

[BENCHMARKING.md]: https://github.com/zlatinb/tiny-classifier/blob/main/BENCHMARKING.md

