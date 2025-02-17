# Tiny Classifier - a configurable neural net classifier

This project aims to create a small neural network capable of categorizing input.  It is written in pure Python and runs on CPU.  Currently only the forward pass is implemented.

### Installation

Make sure you have OpenBLAS installed if you want to make use of your CPU's vector instructions (AVX/AVX2):

```
pip uninstall numpy
sudo apt install openblas
BLAS=openblas pip install numpy
```

Clone the repo and launch the `forward.py` file.  Currently it takes positional arguments:
```
./forward.py 100 f32 64 512 0.00001 sigmoid
755.64155 ms
```

* arg 1 (100) : how many forward passes to make
* arg 2 (f32) : whether to use single ("f32") or double ("f64") precision
* arg 3 (64) : number of layers in the neural net
* arg 4 (512) : number of perceptrons per layer
* arg 4 (0.00001) : epsilon to be used to ensure numerical stability
* arg 5 ("sigmoid") : which activation function to use.  "tanh" is also available

Output is how long the combined forward passes took.  You can use this to benchmark how your CPU handles parameters.

The goal is to build a tool that can be used to pre-process large unlabaled datasets after training on smaller labeled ones.
