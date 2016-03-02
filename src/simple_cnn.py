#!/usr/bin/env python
"""
@name: Simple CNN
@desc: Simple implementation of a CNN on our data.
@auth: Tim O'Brien
@date: 19 Feb. 2016
"""

# Usual imports
import time
import os
import numpy as np

# Deep learning related
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode #TODO remove
import lasagne

# My modules
from generate_data import get_data, use_preparsed_data

def rel_error(x, y):
    """ Returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8. np.abs(x) + np.abs(y))))

def build_cnn(input_var=None):
    """
    Build the CNN architecture.
    """

    # input layer
    network = lasagne.layers.InputLayer(
        shape=(
            None,
            1,
            20,
            129
            ),
        input_var=input_var
        )

    # conv
    network = lasagne.layers.Conv2DLayer(
        lasagne.layers.batch_norm(network), # Batch norm on incoming
        num_filters=16,    # Number of convolution filters to use
        filter_size=(3, 3),
        stride=(1, 1),     # Stride fo (1,1)
        pad='same',        # Keep output size same as input
        nonlinearity=lasagne.nonlinearities.rectify, # ReLU
        W=lasagne.init.GlorotUniform()   # W initialization
        )

    # conv
    network = lasagne.layers.Conv2DLayer(
        lasagne.layers.batch_norm(network), # Batch norm on incoming
        num_filters=16,    # Number of convolution filters to use
        filter_size=(3, 3),
        stride=(1, 1),     # Stride fo (1,1)
        pad='same',        # Keep output size same as input
        nonlinearity=lasagne.nonlinearities.rectify, # ReLU
        W=lasagne.init.GlorotUniform()   # W initialization
        )

    # pool (2x2 max pool)
    network = lasagne.layers.MaxPool2DLayer(
        network, pool_size=(2, 2)
        )

    # conv
    network = lasagne.layers.Conv2DLayer(
        lasagne.layers.batch_norm(network), # Batch norm on incoming
        num_filters=16,    # Number of convolution filters to use
        filter_size=(3, 3),
        stride=(1, 1),     # Stride fo (1,1)
        pad='same',        # Keep output size same as input
        nonlinearity=lasagne.nonlinearities.rectify, # ReLU
        W=lasagne.init.GlorotUniform()   # W initialization
        )

    # conv
    network = lasagne.layers.Conv2DLayer(
        lasagne.layers.batch_norm(network), # Batch norm on incoming
        num_filters=16,    # Number of convolution filters to use
        filter_size=(3, 3),
        stride=(1, 1),     # Stride fo (1,1)
        pad='same',        # Keep output size same as input
        nonlinearity=lasagne.nonlinearities.rectify, # ReLU
        W=lasagne.init.GlorotUniform()   # W initialization
        )

    # pool (2x2 max pool)
    network = lasagne.layers.MaxPool2DLayer(
        network, pool_size=(2, 2)
        )

    # Fully-connected layer of 256 units with 50% dropout on its inputs
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=128,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.HeUniform()   # W initialization
        )

    # Finally add a 1-unit softmax output layer
    network = lasagne.layers.DenseLayer(
        network,
        num_units=1,
        nonlinearity=lasagne.nonlinearities.softmax
        )

    return network


def iterate_minibatches(datadict, batchsize, shuffle=False):
    """
    Generate a minibatch.
    Arguments: datadict  (dictionary, e.g. an output of get_data()
                          in generate_data.py)
               batchsize (int)
               shuffle   (bool, default=False)
    Returns:   inputs[excerpt], targets[excerpt]
    """


    assert datadict['Xshape'][0] == datadict['yshape'][0]

    n_total = datadict['Xshape'][0] # Total number of data points

    Xpath = os.path.abspath(os.path.join(datadict['datadir'],datadict['Xfile']))
    ypath = os.path.abspath(os.path.join(datadict['datadir'],datadict['yfile']))

    Xshape = datadict['Xshape']
    offsetmul = Xshape[1] * Xshape[2] * Xshape[3]

    if shuffle:
        indices = np.arange(batchsize)
        np.random.shuffle(indices)

    for start_idx in range(0, n_total - batchsize + 1, batchsize):
        inputs = np.memmap(
            Xpath,
            dtype='float32',
            mode='r',
            shape=(batchsize, Xshape[1], Xshape[2], Xshape[3]),
            offset=start_idx*offsetmul
            )
        targets = np.memmap(
            ypath,
            dtype='float32',
            mode='r',
            shape=(batchsize, 1),
            offset=start_idx
            )

        if shuffle:
            excerpt = indices[0:batchsize]
        else:
            excerpt = slice(0, batchsize)
        yield inputs[excerpt], targets[excerpt]

def main(num_epochs=1):
    """
    Main function
    """

    # Theano config
    theano.config.floatX = 'float32'

    train, val, test = None, None, None
    try:
        train, val, test = use_preparsed_data(
            outputdir='./data/',
            )
    except:
        train, val, test = get_data(
            n_songs_train=1,
            n_songs_val=1,
            n_songs_test=1,
            outputdir='./data/',
            seed=None
            )

    # Save the returned metadata
    np.savez('./data/metadata', train, val, test)

    # Print the dimensions
    print "Data dimensions:"
    for datapt in [train['Xshape'], train['yshape'],
                   val['Xshape'], val['yshape'],
                   test['Xshape'], test['yshape']]:
        print datapt

    # Parse dimensions
    n_train  = train['yshape'][0]
    n_val    = val[  'yshape'][0]
    n_test   = test[ 'yshape'][0]
    n_chan   = train['Xshape'][1]
    n_feats  = train['Xshape'][2]
    n_frames = train['Xshape'][3]

    print "n_train  = {0}".format(n_train)
    print "n_val    = {0}".format(n_val)
    print "n_test   = {0}".format(n_test)
    print "n_chan   = {0}".format(n_chan)
    print "n_feats  = {0}".format(n_feats)
    print "n_frames = {0}".format(n_frames)

    # Prepare Theano variables for inputs and targets
    input_var  = T.tensor4( name='inputs' )
    target_var = T.fcol( name='targets' )

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions..."),
    network = build_cnn(input_var)
    print("Done.")

    # Create a loss expression for training, i.e., a scalar objective we want to minimize
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()

    # Create update expressions for training
    # Here, we'll use adam
    params  = lasagne.layers.get_all_params(
        network,
        trainable=True
    )
    updates = lasagne.updates.adam(
        loss,
        params,
        learning_rate=1e-3
    )

    # Create a loss expression for validation/testing.
    # The crucial difference here is that we do a deterministic forward pass
    # through the network, disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    test_loss = lasagne.objectives.squared_error(
        test_prediction,
        target_var
        )
    test_loss = test_loss.mean()

    test_pred_fn = theano.function(
        [input_var],
        test_prediction,
        allow_input_downcast=True
        )

    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function(
        [input_var, target_var],
        loss,
        updates=updates,
        mode=NanGuardMode(                                          #TODO remove
            nan_is_error=True, inf_is_error=True, big_is_error=True #TODO remove
            ),                                                      #TODO remove
        allow_input_downcast=True
    )

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function(
        [input_var, target_var],
        [test_loss, test_acc],
        allow_input_downcast=True
    )

    # Finally, launch the training loop.
    print("Starting training...")

    train_error_hist = []

    # We iterate over epochs:
    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(
            train, 500, shuffle=True
            ):
            inputs, targets = batch
            train_err_increment = train_fn(inputs, targets)
            train_err += train_err_increment
            train_error_hist.append(train_err_increment)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
    print("Done training.")

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    test_predictions = []
    for batch in iterate_minibatches(test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_predictions.append( test_pred_fn(inputs) )
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    timestr = str(time.time())
    np.savez('./data/model'+timestr+'.npz', *lasagne.layers.get_all_param_values(network))
    np.save('./data/train_error_hist'+timestr+'.npy', train_error_hist)
    np.save('./data/test_predictions'+timestr+'.npy', test_predictions)
    print "Wrote model to {0}, test error histogram to {1}, and test predictions to {2}".format(
        'model'+timestr+'.npz',
        'train_error_hist'+timestr+'.npy',
        'test_predictions'+timestr+'.npy'
        )

    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == "__main__":
    main()

