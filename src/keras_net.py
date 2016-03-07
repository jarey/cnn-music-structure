#!/usr/bin/env python
"""
@name: keras_net.py
@desc: CNN implementation of music structure segmentation in Keras.
@auth: Tim O'Brien
@date: Winter 2016
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Import keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

# My functions
import generate_data # my data generation function

# Callback for loss history
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


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

    x_path = os.path.abspath(
        os.path.join(datadict['datadir'], datadict['Xfile'])
        )
    y_path = os.path.abspath(
        os.path.join(datadict['datadir'], datadict['yfile'])
        )

    x_shape = datadict['Xshape']
    offsetmul = x_shape[1] * x_shape[2] * x_shape[3]

    if shuffle:
        indices = np.arange(batchsize)
        np.random.shuffle(indices)

    for start_idx in range(0, n_total - batchsize + 1, batchsize):
        inputs = np.memmap(
            x_path,
            dtype='float32',
            mode='r',
            shape=(batchsize, x_shape[1], x_shape[2], x_shape[3]),
            offset=start_idx*offsetmul
            )
        targets = np.memmap(
            y_path,
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


def main(
        num_epochs=1,
        n_songs_train=1,
        n_songs_val=1,
        n_songs_test=1,
        batch_size=256,
        learning_rate=1e-4,
        datadir=os.path.abspath('../salami-audio/'),
        salamidir=os.path.abspath('../salami-data-public/'),
        outputdir=os.path.abspath('./bindata/'),
        reg_amount=0.01
    ):
    """
    Main function
    """

    # DATA ####################################################################

    train, val, test = None, None, None
    try:
        # Try to load data, if we already serialized it and have a
        # datadicts.npz file available
        train, val, test = use_preparsed_data(
            outputdir=outputdir
            )
    except:
        # Otherwise generate the data ourselves
        train, val, test = generate_data.get_data(
            datadir=datadir,
            salamidir=salamidir,
            outputdir=outputdir,
            n_songs_train=n_songs_train,
            n_songs_val=n_songs_val,
            n_songs_test=n_songs_test,
            seed=None
            )

    # Print the dimensions
    print "Data dimensions:"
    for datadict in [train, val, test]:
        print '\t', datadict['Xshape'], '\t', datadict['yshape']


    # CNN MODEL ###############################################################
    # VGG-like convnet, from Keras examples, http://keras.io/examples/
    model = Sequential()
    model.add(Convolution2D(
        16, 3, 3,
        border_mode='valid',
        input_shape=(1, 128, 129),
        init='glorot_normal',
        W_regularizer=l2(reg_amount),
        b_regularizer=l2(reg_amount)
        ))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(
        16, 3, 3,
        init='glorot_normal',
        W_regularizer=l2(reg_amount),
        b_regularizer=l2(reg_amount)
        ))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(
        16, 3, 3,
        border_mode='valid',
        init='glorot_normal',
        W_regularizer=l2(reg_amount),
        b_regularizer=l2(reg_amount)
        ))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(
        16, 3, 3,
        init='glorot_normal',
        W_regularizer=l2(reg_amount),
        b_regularizer=l2(reg_amount)
        ))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(
        256,
        init='glorot_normal',
        W_regularizer=l2(reg_amount),
        b_regularizer=l2(reg_amount)
        ))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(
        1,
        init='glorot_normal',
        W_regularizer=l2(reg_amount),
        b_regularizer=l2(reg_amount)
        ))
    model.add(Activation('linear'))

    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

    print "Compiling the model...",
    model.compile(loss='msle', optimizer=sgd)
    print "Done."


    # FIT MODEL ###############################################################

    # Callback for model checkpointing
    checkpointer = ModelCheckpoint(
        filepath=os.path.abspath(os.path.join(outputdir, "weights.hdf5")),
        verbose=1,
        save_best_only=True
        )

    hist = model.fit_generator(
        iterate_minibatches(train, batchsize, shuffle=True),
        train['Xshape'][0]/2, # samples per epoch - half the train set
        num_epochs,
        callbacks=[checkpointer, history],
        validation_data = iterate_minibatches(val, batchsize, shuffle=True),
        nb_val_samples = val['Xshape'][0],
        nb_worker=3,
        nb_val_worker=3
        )

    print "Fit history"
    print hist.history

    # SAVE SOME PLOTS
    plt.figure()
    plt.plot(history.losses)
    plt.xlabel('Minibatch')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.savefig(
        os.path.abspath(os.path.join(outputdir, 'loss_history_train.pdf')),
        bbox_inches='tight'
        )

if __name__ == "__main__":
    P = argparse.ArgumentParser(
        description='Run a CNN for music structure segmentation.'
        )
    P.add_argument(
        '-t', '--train',
        help='Number of songs to include in training set.',
        required=False,
        default=1
        )
    P.add_argument(
        '-v', '--val',
        help='Number of songs to include in validation set.',
        required=False,
        default=1
        )
    P.add_argument(
        '-s', '--test',
        help='Number of songs to include in test set.',
        required=False,
        default=1
        )
    P.add_argument(
        '-e', '--epochs',
        help='Number of epochs to run.',
        required=False,
        default=1
        )
    P.add_argument(
        '-l', '--learningrate',
        help='Learning rate, for update.',
        required=False,
        default=1e-3
        )
    P.add_argument(
        '-b', '--batchsize',
        help='Batch size.',
        required=False,
        default=256
        )
    P.add_argument(
        '-d', '--datadir',
        help='Directory with salami audio files.',
        required=False,
        default='/user/t/tsob/Documents/cs231n/proj/data/'
        )
    P.add_argument(
        '-ds', '--salamidir',
        help='Directory with salami annotation files.',
        required=False,
        default='/usr/ccrma/media/databases/mir_datasets/salami/salami-data-public/'
        )
    P.add_argument(
        '-w', '--workingdir',
        help='Directory for intermediate data and model files.',
        required=False,
        default='/zap/tsob/audio/'
        )
    P.add_argument(
        '-r', '--regamount',
        help='Regularization strength, for both W and b weights.',
        required=False,
        default=0.01
        )

    ARGS = P.parse_args()

    # Start the show
    main(
        num_epochs=int(ARGS.epochs),
        n_songs_train=int(ARGS.train),
        n_songs_val=int(ARGS.val),
        n_songs_test=int(ARGS.test),
        learning_rate=float(ARGS.learningrate),
        batch_size=int(ARGS.batchsize),
        datadir=os.path.abspath(ARGS.datadir),
        salamidir=os.path.abspath(ARGS.salamidir),
        outputdir=os.path.abspath(ARGS.workingdir),
        reg_amount=float(ARGS.regamount)
        )
