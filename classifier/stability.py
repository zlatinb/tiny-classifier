import numpy as np
from . import *

class Stabilizer :

    def __init__(self, props) :
        self.__dtype = props.dtype
        finfo = np.finfo(props.dtype)
        self.__clip_exp_min = finfo.min + props.epsilon_exp
        self.__clip_exp_max = np.log(finfo.max, dtype=props.dtype) - props.epsilon_exp
        self.__clip_square_min = props.epsilon_square - np.sqrt(finfo.max - props.epsilon_square)
        self.__clip_square_max = np.sqrt(finfo.max) - props.epsilon_square

    def clip_exp(self, x) :
        np.clip(x, a_min = self.__clip_exp_min, \
            a_max = self.__clip_exp_max, \
            out = x, \
            dtype = self.__dtype)

    def clip_square(self, x):
        np.clip(x, a_min = self.__clip_square_min, \
            a_max = self.__clip_square_max, \
            out = x, \
            dtype = self.__dtype)
