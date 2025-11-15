""" 			  		 			     			  	   		   	  			  	
MLP Model.  (c) 2021 Georgia Tech

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

# Do not use packages that are not in standard distribution of python
import numpy as np

np.random.seed(1024)
from ._base_network import _baseNetwork


class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()

    def _weight_init(self):
        """
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        """

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        """
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        """
        loss = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################
        Z_1 = X.dot(self.weights['W1']) + self.weights['b1']
        A = self.sigmoid(Z_1)
        Z_2 = A.dot(self.weights['W2']) + self.weights['b2']
        P = self.softmax(Z_2)
        N = X.shape[0]
        loss = self.cross_entropy_loss(P, y)
        accuracy = self.compute_accuracy(P, y)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################
        
        # dl_dw2 = dl_dp * dp_dz2 * dz2_dw2
        # dl_db2 = sum(dl_dp * dp_dz2)
        # dl_dw1 = dl_dw2 * dw2_da * da_dz1 * dz1_dw1
        # dl_db1 = sum(dl_dw2 * dw2_da * da_dz1)
        
        #dl_dz2 = dl_dp * dp_dz2
        dl_dp = P
        for i in range(N):
            dl_dp[i, y[i]] -= 1
        dl_dz2 = dl_dp / N

        # dl_dW2 = dl_dz2 * dz2_dw2
        # dl_db2 = sum(dl_dz2)
        dl_dW2 = A.T.dot(dl_dz2)
        dl_db2 = dl_dz2.sum(axis=0)

        # dl_da = dl_dz2 * dz2_da
        dl_da = dl_dz2.dot(self.weights['W2'].T)

        # dl_dz1 = dl_da * da_dz1
        dl_dz1 = dl_da * self.sigmoid_dev(Z_1)

        # dl_dW1 = dl_dz1 * dz1_dw1
        # dl_db1 = sum(dl_dz1)
        dl_dW1 = X.T.dot(dl_dz1)
        dl_db1 = dl_dz1.sum(axis=0)

        self.gradients['W1'] = dl_dW1
        self.gradients['b1'] = dl_db1
        self.gradients['W2'] = dl_dW2
        self.gradients['b2'] = dl_db2

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, accuracy
