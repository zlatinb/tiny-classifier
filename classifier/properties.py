import numpy as np

class GlobalProperties :

    def __init__(self, dtype, epsilon):
        if dtype == "f32" :
            dtype = np.float32
        else :
            dtype = np.float64
        self.dtype = dtype
        self.epsilon = dtype(epsilon)
        self.f_1 = dtype(1.0)
        

