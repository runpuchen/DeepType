import os
import argparse
import numpy as np
#import pandas as pd

import tensorflow as tf


######################################################################################################################

class AutoEncoder(object):

    _weights_str = "weights{0}"
    _biases_str = "biases{0}"

    def __init__(self, shape):
        '''Autoencoder initializer

        Args:
        shape: list of ints specifying
              num input, hidden1 units,...hidden_n units, num logits
        sess: tensorflow session object to use
        '''
        self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]
        self.__num_hidden_layers = len(self.__shape) - 2

        self.__variables = {}


        self._setup_variables()


    @property
    def shape(self):
        return self.__shape

    @property
    def num_hidden_layers(self):
        return self.__num_hidden_layers



    def __getitem__(self, item):
        """Get autoencoder tf variable

        Returns the specified variable created by this object.
        Names are weights#, biases#, biases#_out, weights#_fixed,
        biases#_fixed.

        Args:
        item: string, variables internal name
        Returns:
        Tensorflow variable
        """
        return self.__variables[item]

    def __setitem__(self, key, value):
        """Store a tensorflow variable

        NOTE: Don't call this explicity. It should
        be used only internally when setting up
        variables.

        Args:
        key: string, name of variable
        value: tensorflow variable
        """
        self.__variables[key] = value

    def _copy_from_other(self, ae_other):
        for i in range(self.__num_hidden_layers + 1):
            # Train weights
            name_w = self._weights_str.format(i + 1)
            self[name_w] = ae_other[name_w]
            # Train biases
            name_b = self._biases_str.format(i + 1)
            self[name_b] = ae_other[name_b]


    def _copy_from_initialize(self, ae_layer_list):

        for i in range(self.__num_hidden_layers + 1):
            # Train weights
            name_w = self._weights_str.format(i + 1)
            self[name_w] = ae_layer_list[name_w]
            # Train biases
            name_b = self._biases_str.format(i + 1)
            tmp = np.reshape(ae_layer_list[name_b], (1, -1))
            print(tmp[0].shape)
            self[name_b] = tmp[0]




    def get_variables_to_init(self, n):
        """Return variables that need initialization

        Args:
        n: initialize to the n-th layer
        """
        assert n > 0
        assert n <= self.__num_hidden_layers + 1

        vars_to_init = [self._w(n), self._b(n)]

        return vars_to_init


    def _setup_variables(self, initialize = False, initialize_data = []):
        for i in range(0, self.__num_hidden_layers + 1):
            # Trainable weights
            name_w = self._weights_str.format(i + 1)
            w_shape = (self.__shape[i], self.__shape[i + 1])
            if initialize:
                w_init = tf.Variable(initialize_data[name_w])
            else:
                w_init = tf.Variable(tf.random_normal(w_shape, stddev=0.02))


            self[name_w] = tf.Variable(w_init, name=name_w, trainable=True)

            # Trainable biases
            name_b = self._biases_str.format(i + 1)
            b_shape = (self.__shape[i + 1],)
            if initialize:
                tmp = np.reshape(initialize_data[name_b], (1, -1))
                b_init = tf.Variable(tmp[0])
            else:
                b_init = tf.Variable(tf.random_normal(b_shape,  stddev=0.02))
            self[name_b] = tf.Variable(b_init, name=name_b, trainable=True)


    def _w(self, n, suffix=""):
        return self[self._weights_str.format(n) + suffix]

    def _b(self, n, suffix=""):
        return self[self._biases_str.format(n) + suffix]

    @staticmethod
    def _activate(x, w, b,  is_target = False):
        if is_target == False:
            y = tf.nn.sigmoid(tf.add(tf.matmul(x, w), b)) # sigmoid, can change to ReLU
            y = tf.maximum(y, 1.e-9)
            y = tf.minimum(y, 1 - 1.e-9)

        else:
            y = tf.nn.softmax(tf.add(tf.matmul(x, w), b))
        return y


    def supervised_net(self, input_pl, n):
        """Get the supervised fine tuning net

        Args:
        input_pl: tf placeholder for ae input data
        Returns:
        Tensor giving full ae net
        """

        last_output = input_pl
        assert n > 0
        assert n <= self.__num_hidden_layers + 1

        for i in range(0, n):
            w = self._w(i + 1)
            b = self._b(i + 1)
            if i < self.__num_hidden_layers:
                last_output = self._activate(last_output, w, b)
            else:
                last_output = self._activate(last_output, w, b, is_target= True)

        return last_output


