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

#
import threading

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


TRAIN_CAP = None

class DataGenerator(object):
    '''
    Generate minibatches from serialized data.
    '''
    def __init__(self, datadict, batch_size=32, shuffle=False, seed=None):
        self.lock = threading.Lock()
        self.data       = datadict
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.seed       = seed

    def next(self):
        # for python 2.x
        # Keep under lock only the mechainsem which advance the indexing of each batch
        # see # http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.flow_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        offsetmul = this.datadict['Xshape'][1] * this.datadict['Xshape'][2] * this.datadict['Xshape'][3]
        x_path = os.path.abspath(
            os.path.join(this.datadict['datadir'], this.datadict['Xfile'])
            )
        y_path = os.path.abspath(
            os.path.join(this.datadict['datadir'], this.datadict['yfile'])
            )
        bX = np.memmap(
            x_path,
            dtype='float32',
            mode='r',
            shape=(current_batch_size, this.datadict['Xshape'][1], this.datadict['Xshape'][2], this.datadict['Xshape'][3]),
            offset=current_index*offsetmul
            )
        bY = np.memmap(
            y_path,
            dtype='float32',
            mode='r',
            shape=(current_batch_size, 1),
            offset=current_index
            )
        return bX, bY

    def flow(self, datadict, batch_size=32, shuffle=False, seed=None,
             save_to_dir=None, save_prefix="", save_format="jpeg"):
        assert datadict['Xshape'] == datadict['yshape']
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.flow_generator = self._flow_index(datadict['Xshape'][0], batch_size, shuffle, seed)
        return self

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):

        # Check cap
        if TRAIN_CAP:
            N = min(N, TRAIN_CAP)

        b = 0
        total_b = 0
        while 1:
            if b == 0:
                if seed is not None:
                    np.random.seed(seed + total_b)

                if shuffle:
                    index_array = np.random.permutation(N)
                else:
                    index_array = np.arange(N)

            current_index = (b * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
            else:
                current_batch_size = N - current_index

            if current_batch_size == batch_size:
                b += 1
            else:
                b = 0
            total_b += 1
            yield index_array[current_index: current_index + current_batch_size], current_index, current_batch_size



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

    history = LossHistory()


    train_batch_gen = DataGenerator(
        train,
        batch_size=batch_size,
        shuffle=True,
        seed=None
        )

    hist = model.fit_generator(
        train_batch_gen,
        min(TRAIN_CAP, train['Xshape'][0]), # samples per epoch
        num_epochs,
        callbacks=[checkpointer, history],
        # validation_data = iterate_minibatches(val, batch_size, shuffle=True),
        # nb_val_samples = val['Xshape'][0],
        nb_worker=3,
        # nb_val_worker=3
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
    P.add_argument(
        '-c', '--traincap',
        help='Maximum number of training examples.',
        required=False,
        default=None
        )

    ARGS = P.parse_args()
    TRAIN_CAP = int(ARGS.traincap)

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
