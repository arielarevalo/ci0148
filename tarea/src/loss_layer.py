from .funcs import *
from .op import *
import numpy as np

#implements a log loss layer
class loss_layer(op):

    def __init__(self, i_size, o_size):
        super(loss_layer, self).__init__(i_size, o_size)
        self.grads = np.zeros((o_size, i_size))

    def forward(self, x):
        self.x = x
        self.o = np.dot(self.x, self.W) + self.b
        return self.o

    #alpha is used as reward in some reinforcement learning envs
    def backward(self, y, rewards=None):
        if rewards is not None:
            self.grads = 2 * (self.o - y) * rewards
        else:
            self.grads = 2 * (self.o - y)

    # MSE loss function
    def loss(self, y):
        return np.mean(np.square(self.o - y))

