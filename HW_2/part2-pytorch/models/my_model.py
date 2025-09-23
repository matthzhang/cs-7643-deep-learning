
"""
MyModel model.  (c) 2021 Georgia Tech

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

import torch
import torch.nn as nn


def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from my_model.py!")


class MyModel(nn.Module):
    """
    A stronger CIFAR-10 baseline:
    - 3x3 padded convs keep spatial size (32x32) early on
    - BatchNorm + ReLU after each conv
    - Downsample only via MaxPool (twice) to preserve information
    - Dropout for regularization
    - Global Average Pooling to reduce parameters and overfitting
    """

    def __init__(self):
        super().__init__()
        #############################################################################
        # Improved initialization and architecture for CIFAR-10                      #
        #############################################################################
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(32)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 4 * 4, 10)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        #############################################################################
        # Forward pass                                                              #
        #############################################################################
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        outs = self.fc(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
