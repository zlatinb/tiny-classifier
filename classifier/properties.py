import numpy as np
from . import *
class GlobalProperties :

    def __init__(self, args):
        if args.precision == "f4" :
            dtype = np.float32
        else :
            dtype = np.float64
        self.dtype = dtype
        self.epsilon_exp = dtype(args.epsilon_exp)
        self.epsilon_square = dtype(args.epsilon_square)
        self.f_1 = dtype(1.0)
        self.activation = args.activation
        self.size = args.dimension
        

