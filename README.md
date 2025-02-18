# Tiny Classifier - a configurable neural net classifier

This project aims to create a small neural network capable of categorizing input.  It is written in pure Python and runs on CPU.  Currently only the forward pass is implemented.  This tool may eventually be used to label large datasets after training on smaller labeled ones.

### Installation


Clone the repo and launch the `benchmark.py` file:
```
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
```
Here are the results from my laptop:
```
./benchmark.py -p f32 -n 20 -l 100 -d 50 -a sigmoid -e 0.0001
Total time for 20 run(s) was 22.169502 ms
```

You can use this to benchmark how your CPU handles different parameters.  If you install OpenBLAS you may be able to take advantage of multi-threaded vector instructions like AVX/AVX2/etc:

```
pip uninstall numpy
sudo apt install openblas
BLAS=openblas pip install numpy
```
