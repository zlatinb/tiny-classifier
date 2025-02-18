# Tiny Classifier - a configurable neural net classifier

This project aims to create a small neural network capable of categorizing input.  It is written in pure Python and runs on CPU.  Currently only the forward pass is implemented.  This tool may eventually be used to label large datasets after training on smaller labeled ones.

### Installation


Clone the repo and launch the `benchmark.py` file:
```bash
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
```bash
./benchmark.py -p f32 -n 20 -l 100 -d 50 -a sigmoid -e 0.0001
Total time for 20 run(s) was 22.169502 ms
```
### Benchmarking

You can use this tool to benchmark how your CPU handles different parameters.  Here are a few things to take into account:

* You can rebuild OpenBLAS to tailor it to your architure.  You will need to reinstall numpy with the `--no-binary` flag after that and set several environment variables.
* Check `numpy.__config__.show()` to see which instructions are available to openblas.  More == better :)
* The result of the first epoch will often be worse, you may want to disregard it.
* Be mindful of your CPU's cache hierarchy.  The more layers you can fit into faster cache levels the better.  A single layer should ideally take no more than 50%-70% of the target cache level.  Since all hidden layers are currently the same size, you can use this formula to compute the size of a layer:
    ```
    size = T * N ^ (N + 2)
    where N is the layer dimension (-d, --dimension) and T is the data type (4 for f32, 8 for f64)
    ```

