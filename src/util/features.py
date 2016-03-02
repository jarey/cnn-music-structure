#!/usr/bin/env python
"""
@name: features.py
@desc: Utility functions for importing audio and calculating features.
@auth: Tim O'Brien
@date: Feb. 16, 2016
"""

import numpy as np
import subprocess
import librosa
import rwc
import ia
import evaluation as ev

VALID_SIDS = rwc.get_ids() + ia.get_ids()

FRAME_SIZE = 1024 # Size of frames, in samples
N_FRAMES_CONTEXT = 33 # 16 frames of context on either side
FS = 44100 # assume sample rate of 44100 Hz. (We know this to be true for our data)
N_MFCC = 13

def id2audio(sid, centered_at=None, context=None):
    """
    Returns the audio and samplerate of a file with given SALAMI ID.

    Arguments:
    sid : the SALAMI id number
    centered_at (optional): time in seconds which the loaded audio chunk will
                            be centered around.
    context (optional) : the duration of our audio chunk, if we're getting a
                         snippet centered somewhere.
    """
    wav_filename = None
    if sid in rwc.get_ids():
        wav_filename = rwc.id2path(sid)
    elif sid in ia.get_ids():
        wav_filename = ia.id2path(sid)
    else:
        return None

    # Do we need to pad with zeros, before start or after end?
    prepadtime = None

    offset = 0.0
    duration = None

    # If we only want a chunk of audio...
    if centered_at:
        offset = centered_at - context/2.0
        duration = context
        if offset < 0.0: # Do we need to pad the beginning with silence?
            prepadtime = -offset # Keep track of padded time
            offset = 0.0
            duration -= prepadtime

    # We want to load in at the correct sample rate. Check with sox.
    fs = int(subprocess.check_output(['soxi', '-r', wav_filename]))

    print offset, duration

    # Do the actual loading...
    y, fs = librosa.load(
        wav_filename,
        sr=FS,
        mono=True,
        offset=offset,
        duration=duration
        )

    if prepadtime: # do we need to pad?
        y = np.hstack([np.zeros(int(prepadtime*fs)), y])
        print "prepad"

    # If we need to post-pad, our file length will be less than we expect.
    if centered_at:
        if y.shape[0] < context*fs:
            postpadsamples = int(context*fs - y.shape[0])
            y = np.hstack([y, np.zeros(postpadsamples)])
            print "postpad"

    return y, fs

def id2mfcc(sid, n_mfcc=13):
    """
    Returns MFCCs for the sid.
    """
    y, fs = id2audio(sid)
    metadata = {}
    metadata['sid'] = sid
    metadata['fs'] = fs
    metadata['n_mfcc'] = 13
    y_mfcc = librosa.feature.mfcc(y, sr=fs, n_mfcc=n_mfcc)
    return y_mfcc, metadata

def get_random_sids(num=1):
    """
    Get a SALAMI id (or more, given the number as argument)
    for a file whose audio we have.
    """
    if num == 1:
        return np.random.choice(VALID_SIDS)
    else:
        return np.random.choice(VALID_SIDS, num)

def get_random_seg():
    """
    Get a randomly selected segmentation time from a random SALAMI track.
    This can be used to generate "positive" segmentation examples.
    """
    sid = get_random_sids()
    times = ev.id2segtimes(sid)
    return sid, np.random.choice(times)

def get_random_positive_audio():
    """
    Get a set of MFCC frames from a randomly selected positive example of a
    segmentation boundary.

    Arguments:
    N : number of frames to use.
    """
    sid, segtime = get_random_seg()
    print sid, segtime
    y, fs = id2audio(
        sid,
        centered_at=segtime,
        context=FRAME_SIZE*N_FRAMES_CONTEXT/float(FS)
        )

    # Print an SOS if we don't have the right FS
    if fs != FS:
        print "ERROR: sample rate is not what I expected for SID {0}.".format(
                sid
                )

    return y, fs


def get_random_positive_MFCCs():
    """
    Get a set of MFCCs centered about a randomly chosen segment boundary.
    """
    y, fs = get_random_positive_audio()
    MFCCs = librosa.feature.mfcc(
        y,
        sr=fs,
        n_mfcc=N_MFCC,
        n_fft=FRAME_SIZE,
        hop_length=FRAME_SIZE/2
        )
    return MFCCs, y, fs

if __name__ == "__main__":
    pass
