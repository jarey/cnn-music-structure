#!/usr/bin/env python
"""
@name: plot.py
@desc: Utility functions for plotting/
@auth: Tim O'Brien
@date: Feb. 16, 2016
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def segplot(y, fs, segments=None):
    fig, ax = plt.subplots()
    plt.plot(np.arange(y.shape[0])/float(fs), y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    if segments is not None:
        for segment in segments:
            ax.add_patch(
                patches.Rectangle(
                    (segment[0],0), # (x,y)
                    segment[1]-segment[0], # width
                    1.0 #height
                    )
                )
    plt.show()


if __name__ == "__main__":
    from rwc import get_ids
    from features import id2audio
    sid = get_ids()[0]
    y, fs = id2audio(sid)
    segplot(y, fs)
