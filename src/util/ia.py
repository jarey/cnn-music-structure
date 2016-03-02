#!/usr/bin/env python
"""
@name: ia.py
@desc: Utility functions to retrieve audio in the IA dataset.
@auth: Tim O'Brien, Feb, 16 2016
"""
import os
import datapaths as dpath
import csv

def get_ids():
    """
    Get all SALAMI IDs related to IA
    """
    # Filename for SALAMI IA metadata
    metadata_file = os.path.join(
        dpath.SALAMI, 'metadata', 'id_index_internetarchive.csv')

    ids = []

    with open(metadata_file, "r") as rwc_file:
        reader = csv.reader(rwc_file)
        next(reader) #skip header
        for row in reader:
            ids.append(int(row[0]))

    return ids

def id2path(sid):
    """
    Get the path to the .wav file corresponding to the SALAMI id number.
    Note: IA only.

    Argument: sid, the salami id number (numeric)
    """

    wav_file_path = os.path.join(
        dpath.IA,                 # Directory of IA mp3 files.
        '{0:d}.mp3'.format(sid)   # Named by SALAMI ID number.
        )

    # Make sure it exists first
    if os.path.exists(wav_file_path):
        return wav_file_path
    else:
        return None


if __name__ == "__main__":
    MYSID = 1000
    print "Salami id = {0}.".format(MYSID)
    print "Wav file path is:\n\t{0}".format(id2path(MYSID))

    print "There are {0:d} ids from IA.".format(len(get_ids()))
