"""
Author: Sophia Sanborn, Sagnik Bhattacharya
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas, github.com/sagnibak
"""

import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from collections import OrderedDict

from typing import Callable, List, Literal, Tuple, Union


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
        self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
    kernel_shape: Tuple[int, int] = None,
    stride: int = None,
    pad: int = None,
    mode: str = None,
    keep_dim: str = "first",
) -> Layer:
    """Factory function for layers."""
    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out, activation=activation, weight_init=weight_init,
        )

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        # Initialize weights and bias
        W = self.init_weights((self.n_in, self.n_out))  # W has shape (n_in, n_out)
        b = np.zeros((1, self.n_out))  # Bias is initialized to zeros with shape (1, n_out)

        # Store parameters and initialize cache and gradients
        self.parameters = OrderedDict({"W": W, "b": b})  # store W and b
        self.cache = OrderedDict()  # cache for storing forward pass values
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)})  # gradients initialized to zeros

        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###
        
        # perform an affine transformation and activation

        # Affine transformation (W * X + b)
        Z = X @ self.parameters["W"] + self.parameters["b"]

        # Apply activation function
        out = self.activation.forward(Z)
        
        # store information necessary for backprop in `self.cache`
        self.cache["X"] = X  # input
        self.cache["Z"] = Z  # pre-activation output

        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  gradient of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###
        
        # unpack the cache
        X = self.cache["X"]
        Z = self.cache["Z"]

        # Derivative of the activation function
        dLdZ = self.activation.backward(Z, dLdY)

        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer

        # Compute the gradients of weights, bias, and input
        dW = np.dot(X.T, dLdZ)  # Gradient of the loss w.r.t. W
        db = np.sum(dLdZ, axis=0, keepdims=True)  # Gradient of the loss w.r.t. b
        dX = np.dot(dLdZ, self.parameters['W'].T)  # Gradient of the loss w.r.t. input X

        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.

        # Store the gradients for weights and bias
        self.gradients['W'] = dW
        self.gradients['b'] = db
        ### END YOUR CODE ###

        return dX
