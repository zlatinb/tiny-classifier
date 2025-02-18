import numpy as np
from . import *
class GlobalProperties :

    def __init__(self, args):
        if args.precision == "f32" :
            dtype = np.float32
        else :
            dtype = np.float64
        self.dtype = dtype
        self.epsilon = dtype(args.epsilon)
        self.f_1 = dtype(1.0)
        self.activation = args.activation
        self.size = args.dimension
        

