#!/usr/bin/env python
"""
@name: generate_data.py
@desc: generate some data
@auth: Tim O'Brien
@date: Feb. 18th, 2016
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import util.evaluation as ev

# Debug?
DEBUG_PLOT = False

# Set some params
FS = 44100 # Enforce 44.1 kHz sample rate
N_FFT = 2048
HOP_LENGTH = N_FFT/2 # 50% overlap
N_MFCC = 13
N_MEL = 128

T_CONTEXT = 3 # seconds of context for our features
N_FRAME_CONTEXT = librosa.time_to_frames(
    T_CONTEXT,
    sr=FS, hop_length=HOP_LENGTH, n_fft=N_FFT
    )[0]+1
# 64 frames on either side, for context

BOUNDARY_KERNEL = signal.gaussian(N_FRAME_CONTEXT, std=32) # For smoothing our y
#BOUNDARY_KERNEL = np.ones(N_FRAME_CONTEXT)

DTYPE = 'float32'

# FOR USE ON AMAZON EC2 AFTER COPYING FROM S3
#DATADIR = os.path.abspath(os.path.join('/mnt','audio'))
#SALAMIDIR = os.path.abspath(os.path.join('/mnt','salami', 'salami-data-public'))

# FOR USE ON CCRMA NETWORK
DATADIR = os.path.abspath('/user/t/tsob/Documents/cs231n/proj/data')
SALAMIDIR = os.path.abspath('/usr/ccrma/media/databases/mir_datasets/salami/salami-data-public')

def get_sids(datadir=DATADIR):
    """
    Get the SALAMI IDS and their audio filenames, returned as lists, for a
    given directory containing all the audio files.
    """
    sids = [int(os.path.splitext(listitem)[0]) \
            for listitem in os.listdir(datadir)]
    paths = [listitem for listitem in os.listdir(datadir)]
    return sids, paths

def get_data(
        datadir=DATADIR,
        salamidir=SALAMIDIR,
        n_songs_train=1,
        n_songs_val=1,
        n_songs_test=1,
        outputdir='/zap/tsob/audio/',
        seed=None
    ):
    """
    Give me some data!
    """
    # Get the salami ID numbers
    sids, paths = get_sids(datadir=datadir)

    # Randomize on a per song basis................
    # Seed, if not None, allows reproducible results
    np.random.seed(seed)
    # Randomize indices so sids and paths still have corresponding indices.
    rand_song_idx = np.random.permutation(len(sids)).tolist()
    # Randomize
    sids = np.asarray(sids)[rand_song_idx]
    paths = np.asarray(paths)[rand_song_idx]

    if (n_songs_train + n_songs_val + n_songs_test) > len(sids):
        print "ERROR: {0} train + {1} val + {2} test exceeds total number of \
                songs in corpus ({3}).".format(
                    n_songs_train,
                    n_songs_val,
                    n_songs_test,
                    len(sids)
                    )
        return # Not enough songs, KTHXBAI

    sids_train = sids[:n_songs_train]
    sids_val = sids[n_songs_train:n_songs_train+n_songs_val]
    sids_test = sids[n_songs_train+n_songs_val:n_songs_train+n_songs_val+n_songs_test]
    paths_train = paths[:n_songs_train]
    paths_val = paths[n_songs_train:n_songs_train+n_songs_val]
    paths_test = paths[n_songs_train+n_songs_val:n_songs_train+n_songs_val+n_songs_test]

    # Put all the output into dictionaries
    train, val, test = {}, {}, {}

    print "Getting training data"
    train['Xfile'], train['Xshape'], train['yfile'], train['yshape'], \
        train['frame_start'] = serialize_data_chunk(
            sids_train,
            paths_train,
            datadir=datadir,
            salamidir=salamidir,
            outputdir=outputdir,
            prefix='train'
            )

    print "Getting validation data"
    val['Xfile'], val['Xshape'], val['yfile'], val['yshape'], \
        val['frame_start'] = serialize_data_chunk(
            sids_val,
            paths_val,
            datadir=datadir,
            salamidir=salamidir,
            outputdir=outputdir,
            prefix='val'
            )

    print "Getting test data"
    test['Xfile'], test['Xshape'], test['yfile'], test['yshape'], \
        test['frame_start'] = serialize_data_chunk(
            sids_test,
            paths_test,
            datadir=datadir,
            salamidir=salamidir,
            outputdir=outputdir,
            prefix='test'
            )

    train['sids'] = sids_train
    train['paths'] = paths_train
    train['audiodir'] = datadir
    train['datadir'] = outputdir

    val['sids'] = sids_val
    val['paths'] = paths_val
    val['audiodir'] = datadir
    val['datadir'] = outputdir

    test['sids'] = sids_test
    test['paths'] = paths_test
    test['audiodir'] = datadir
    test['datadir'] = outputdir

    # Save the dicts for later
    np.savez(
        os.path.join(outputdir,'datadicts.npz'),
        train=train,
        val=val,
        test=test
        )

    return train, val, test

def use_preparsed_data(outputdir='/zap/tsob/audio/'):
    """
    Give me some data that I already computed with get_data()!
    """
    npzfile = np.load(os.path.join(outputdir,'datadicts.npz'))
    return npzfile['train'].tolist(), npzfile['val'].tolist(), npzfile['test'].tolist()

def serialize_data_chunk(
        sids,
        paths,
        datadir=DATADIR,
        salamidir=SALAMIDIR,
        outputdir='.',
        prefix='data'
        ):
    """
    serialize_data_chunk()
    Serializes a chunk of data on disk, given SIDs and corresponding paths.

    Arguments:
        sids      :  the SIDs (int list)
        paths     :  paths to sids audio files (string list)
        datadir   : where the audio files are stored
        salamidir : i.e. the salami-data-public dir from a cloned SALAMI repo
        outputdir : for serialized data on disk
        prefix    : prefix for serialized data file on disk

    Outputs:
        X_path  : string paths to the serialized files
        X_shape : shape of data serialized in X_path
        y_path  : string paths to the serialized files
        y_shape : shape of data serialized in y_path
    """

    timestr = str(time.time())

    X, y = None, None
    X_path, X_shape, y_path, y_shape = None, None, None, None

    offset_X = 0
    offset_y = 0

    frame_start = [0]

    X_shape = [0, 1, N_MEL, N_FRAME_CONTEXT]
    y_shape = [0, 1]

    # Get data song by song
    for i in xrange(len(sids)):

        print "SID: {0},\tfile: {1}".format(sids[i], paths[i])

        # Get the annotated segment times (sec)
        times = ev.id2segtimes(
            sids[i],
            ann_type="uppercase",
            salamipath=salamidir
            )
        times_frames = librosa.time_to_frames(
            times,
            sr=FS,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT
            )

        # Get signal
        sig, fs = librosa.load(
            os.path.join(DATADIR, paths[i]),
            FS
            )

        # Get feature frames
        sig_feat = librosa.feature.melspectrogram(
            y=sig,
            sr=fs,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MEL,
            fmax=1600
            )
        sig_feat = 20.0*np.log10( sig_feat ) # convert to dB
        sig_feat = sig_feat - np.max(sig_feat)

        # debug plot
        if(DEBUG_PLOT):
            plt.figure()
            plt.imshow(
                sig_feat,
                origin='lower',
                interpolation='nearest',
                aspect='auto'
                )
            plt.colorbar()
            plt.show()

        # Keep track of the number of frames for this song
        n_frames = sig_feat.shape[1]

        y_shape[0] += n_frames # increment the shape of our final output y data
        X_shape[0] += n_frames # increment the shape of our final output y data

        # Pad the frames, so we can have frames centered at the very start and
        # end of the song.
        sig_feat = np.hstack((
            np.zeros((N_MEL, N_FRAME_CONTEXT/2)),
            sig_feat,
            np.zeros((N_MEL, N_FRAME_CONTEXT/2))
            ))

        # Generate the boundary indicator
        y_path = os.path.abspath(os.path.join(outputdir, prefix + timestr + '_y'))
        yi = np.memmap(
            y_path,
            dtype=DTYPE,
            mode='w+',
            shape=(n_frames, 1),
            offset=offset_y
            )
        yi[:] = np.zeros((n_frames,1))[:] # start with zeros
        yi[np.minimum(times_frames,n_frames-1),0] = 1.0

        # Smooth y with the gaussian kernel
        yi[:,0] = np.convolve( yi[:,0], BOUNDARY_KERNEL, 'same')
        yi[:,0] = np.minimum(yi[:,0],1.0) # nothing above 1

        # Generate the training data
        X_path = os.path.abspath(os.path.join(outputdir, prefix + timestr + '_X'))
        Xi = np.memmap(
                X_path,
                dtype=DTYPE,
                mode='w+',
                shape=(n_frames, 1, N_MEL, N_FRAME_CONTEXT),
                offset=offset_X
                )

        # Append the start frame for our next song
        if (i != len(sids)-1):
            frame_start.append(frame_start[-1]+Xi.shape[0])

        for i_frame in xrange(n_frames):
            Xi[i_frame,0] = sig_feat[:,i_frame:i_frame+N_FRAME_CONTEXT]

        offset_X += Xi.size
        offset_y += yi.size

        # debug plot
        if(DEBUG_PLOT):
            plt.figure()
            plt.subplot(211)
            plt.imshow(Xi[Xi.shape[0]/2,0])
            plt.colorbar()
            plt.subplot(212)
            plt.plot(yi)
            plt.show()

        # Flush our binary data to file, prepare for next song
        Xi.flush()
        yi.flush()


    return X_path, X_shape, y_path, y_shape, frame_start

if __name__ == "__main__":
    train, val, test = get_data(
        datadir=DATADIR,
        salamidir=SALAMIDIR,
        n_songs_train=5,
        n_songs_val=2,
        n_songs_test=2,
        outputdir='/zap/tsob/audio/',
        seed=None
        )
    print train, val, test
