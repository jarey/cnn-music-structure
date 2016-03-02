#!/usr/bin/env python
"""
@name: datapaths.py
@desc: Contains all the paths to data directories.
@auth: Tim O'Brien
@date: Feb. 16, 2016
"""

import os

MIR = '/usr/ccrma/media/databases/mir_datasets/'
SALAMI = os.path.join(MIR, 'salami', 'salami-data-public')
IA = os.path.join(MIR, 'salami', 'SALAMI-IA')
RWC = os.path.join(MIR, 'RWC')

if __name__ == "__main__":
    pass
