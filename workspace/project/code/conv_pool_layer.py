import theano

from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

import numpy as np


class ConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a ConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))

        w_bound = np.sqrt(6./(fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-w_bound, high=w_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True)

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

        self.input = input
