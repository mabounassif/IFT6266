import theano

from workspace.project.code.conv_pool_layer import ConvPoolLayer
from workspace.project.code.mlp import HiddenLayer
from workspace.project.code.logistic_sgd import LogisticRegression

from theano import tensor as T


class NNet(object):
    def __init__(self, rng, batch_size, data, nkerns_options, learning_rate):
        print('... building the nnet')

        x = T.matrix('x')  # the data is presented as rasterized images
        y = T.ivector('y')

        self.data = data

        index = T.lscalar()

        layer0_input = x.reshape((batch_size,) + nkerns_options[0]['input_shape'])
        layer0 = ConvPoolLayer(
            rng,
            input=layer0_input,
            input_shape=(batch_size,) + nkerns_options[0]['input_shape'],
            filter_shape=nkerns_options[0]['filter_shape'],
            poolsize=nkerns_options[0]['poolsize']
        )

        layer1 = ConvPoolLayer(
            rng,
            input=layer0.output,
            input_shape=(batch_size,) + (nkerns_options[1]['input_shape']),
            filter_shape=nkerns_options[1]['filter_shape'],
            poolsize=nkerns_options[1]['poolsize']
        )

        layer2_input = layer1.output.flatten(2)
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns_options[2]['n_in'],
            n_out=nkerns_options[2]['n_out'],
            activation=T.tanh
        )

        layer3 = LogisticRegression(
            input=layer2.output,
            n_in=nkerns_options[3]['n_in'],
            n_out=nkerns_options[3]['n_out']
        )

        cost = layer3.negative_log_likelihood(y)

        self.test_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x: data['test_set']['x'][index * batch_size: (index + 1) * batch_size],
                y: data['test_set']['y'][index * batch_size: (index + 1) * batch_size]
            }
        )

        self.validate_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x: data['valid_set']['x'][index * batch_size: (index + 1) * batch_size],
                y: data['valid_set']['y'][index * batch_size: (index + 1) * batch_size]
            }
        )

        params = layer3.params + layer2.params + layer1.params + layer0.params

        grads = T.grad(cost, params)

        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
            ]

        self.train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: data['train_set']['x'][index * batch_size: (index + 1) * batch_size],
                y: data['train_set']['y'][index * batch_size: (index + 1) * batch_size]
            }
        )
