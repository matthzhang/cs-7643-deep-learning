"""
2d Convolution Module.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from convolution.py!")

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        N, C, H, W = x.shape
        F = self.out_channels
        HH = WW = self.kernel_size
        s = self.stride
        p = self.padding

        # Output spatial dims (assumes exact fit)
        H_out = 1 + (H + 2 * p - HH) // s
        W_out = 1 + (W + 2 * p - WW) // s

        out = np.zeros((N, F, H_out, W_out), dtype=x.dtype)

        # Correct padding: each axis needs (before, after)
        if p > 0:
            x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        else:
            x_pad = x  # avoid extra copy when no padding

        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    h_start = i * s
                    h_end = h_start + HH
                    for j in range(W_out):
                        w_start = j * s
                        w_end = w_start + WW
                        window = x_pad[n, :, h_start:h_end, w_start:w_end]
                        out[n, f, i, j] = np.sum(window * self.weight[f]) + self.bias[f]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        N, C, H, W = x.shape
        F = self.out_channels
        HH = WW = self.kernel_size
        s = self.stride
        p = self.padding

        # Derived output dims (or take from dout.shape)
        _, _, H_out, W_out = dout.shape

        # Initialize grads
        self.dw = np.zeros_like(self.weight)
        self.db = np.zeros_like(self.bias)

        # Pad x for convenience; also maintain a padded dx to accumulate into
        if p > 0:
            x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
            dx_pad = np.zeros_like(x_pad)
        else:
            x_pad = x
            dx_pad = np.zeros_like(x)

        # Compute gradients
        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    h_start = i * s
                    h_end = h_start + HH
                    for j in range(W_out):
                        w_start = j * s
                        w_end = w_start + WW

                        window = x_pad[n, :, h_start:h_end, w_start:w_end]  # (C, HH, WW)
                        upstream = dout[n, f, i, j]

                        # db: sum over N, i, j
                        self.db[f] += upstream

                        # dw: sum over N, i, j of window * upstream
                        self.dw[f] += window * upstream

                        # dx: distribute filter back into the input window
                        dx_pad[n, :, h_start:h_end, w_start:w_end] += self.weight[f] * upstream

        # Unpad dx if we padded the input
        if p > 0:
            self.dx = dx_pad[:, :, p:-p, p:-p]
        else:
            self.dx = dx_pad
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
