# Benchmarking

### Why bother beyond a quick benchmark

If you intend to use this tool for serious tasks or have a very large dataset, it is a good idea to do more elaborate benchmarking.  The [benchmark.py] tool supports yaml-driven configuration.

Example yaml config:
```yaml
version : 1                             # backward compatibility will be on a best-effort basis
benchmark :
    name : "Hello World!"               # optional
    nepochs : 100                       # in the benchmark 1 epoch == 1 inference
    shape :
        layers : 100                    # only hidden layers, input and output layer excluded
        dimension : 128                 # measured in perceptrons
    math :
        precision : "f32"               # possible values are f32 and f64 on x86, other acrchitectures?
        epsilons :                      # used for numerical stability.  Increase if you start getting overflows
            epsilon_exp : "4e-6"        # for FP32 this is the smallest value I found
            epsilon_square : "1e-7"     # TODO
        activation : "sigmoid"          # available "sigmoid" and "tanh"
    performance :
        threads : 5                     # if not present means unlimited
```

Example usage:
```bash
./benchmark.py -c example_benchmark.yaml
Completed "Hello World!" in 179.538458 ms
```
### Various things to take into account

* You can rebuild OpenBLAS to tailor it to your architure.  You will need to reinstall numpy with the `--no-binary` flag after that and set several environment variables.
* Check `numpy.__config__.show()` to see which instructions are available to openblas.  More == better :)
* The result of the first epoch will often be worse, you may want to disregard it.
* For fewer layers with smaller dimensions using a single thread (`-t 1`) is sometimes faster.
* Be mindful of your CPU's cache hierarchy.  The more layers you can fit into faster cache levels the better.  A single layer should ideally take no more than 50%-70% of the target cache level.  Since all hidden layers are currently the same size, you can use this formula to compute the size of a layer:
    ```
    size = T * N ^ (N + 2)
    where N is the layer dimension (-d, --dimension) and T is the data type (4 for f32, 8 for f64)

### Tuning the math

I will be exposing more hyperparameters and updating this section frequently.

Regarding activation functions, on my cheap laptop (ZEN2) the `tanh` activation function is much slower than `sigmoid`.  Once I implement backpropagation I intend to see which one gives better results.

Choosing the right epsilons is more involved process as it involves doing iterative search at each power until you're satisfied with the forward error.  

[benchmark.py]: https://github.com/zlatinb/tiny-classifier/blob/main/benchmark.py
