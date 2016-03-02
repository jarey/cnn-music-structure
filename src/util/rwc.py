#!/usr/bin/env python
"""
@name: rwc.py
@desc: Utility functions to retrieve audio in the RWC dataset.
@auth: Tim O'Brien, Feb, 16 2016
"""
import os
import csv
import datapaths as dpath

def get_ids():
    """
    Get all SALAMI IDs related to RWC
    """
    # Filename for SALAMI RWC metadata
    metadata_file = os.path.join(
        dpath.SALAMI, 'metadata', 'id_index_rwc.csv')

    ids = []

    with open(metadata_file, "r") as rwc_file:
        reader = csv.reader(rwc_file)
        next(reader) #skip header
        for row in reader:
            ids.append(int(row[0]))

    ids = ids[1:] # First one has no annotations!?

    return ids


def id2path(sid):
    """
    Get the path to the .wav file corresponding to the SALAMI id number.
    Note: RWC only.

    Argument: sid, the salami id number (numeric)
    """

    # Filename for SALAMI RWC metadata
    metadata_file = os.path.join(
        dpath.SALAMI, 'metadata', 'id_index_rwc.csv')

    with open(metadata_file, "r") as rwc_file:
        reader = csv.reader(rwc_file)
        next(reader) #skip header
        for row in reader:
            if int(row[0]) == int(sid):

                filename = os.path.join(
                    dpath.RWC,                 # Main RWC directory
                    'RWC-{0}'.format(row[2]),  # Specific RWC dataset directory
                    "Disc{0:0=2d}-Track{1:0=2d}.wav".format(
                        int(row[3]), int(row[4])
                        )
                    )

                # Make sure the file exists
                if os.path.exists(filename):
                    return filename
                else:
                    return None


if __name__ == "__main__":
    MYSID = 1502
    print "Salami id = {0}.".format(MYSID)
    print "Wav file path is:\n\t{0}".format(id2path(MYSID))

    print "There are {0:d} ids from RWC.".format(len(get_ids()))
