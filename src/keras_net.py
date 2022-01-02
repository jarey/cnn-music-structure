#!/usr/bin/env python
"""
@name: keras_net.py
@desc: CNN implementation of music structure segmentation in Keras.
@auth: Tim O'Brien
@date: Winter 2016
"""

import argparse
import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt

#
import threading

# Import keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

# My functions
from past.builtins import xrange

import generate_data  # my data generation function

# One big list of the valid SALAMI ids
SIDS = [1258, 1522, 1491, 1391, 986, 1392, 1562, 1006, 1303, 1514, 982, 1095,
        1130, 1216, 1204, 1536, 1492, 1230, 1503, 1096, 1220, 996, 976, 1010,
        1120, 1064, 1292, 1008, 1431, 1206, 1074, 1356, 1406, 1559, 1566, 1112,
        1278, 1540, 1423, 1170, 1372, 1014, 1496, 1327, 1439, 1528, 1311, 1226,
        1138, 1016, 1364, 1484, 1338, 1254, 968, 998, 1302, 1075, 1018, 1166,
        1239, 1080, 1032, 1447, 984, 1382, 1284, 1043, 1378, 1467, 1038, 1499,
        1059, 1534, 1283, 1352, 1524, 1428, 1502, 1088, 1236, 1543, 1475, 1551,
        990, 1589, 1282, 1459, 1379, 1542, 1131, 1460, 1050, 1128, 991, 1560,
        1139, 1527, 1270, 1450, 1348, 1331, 1091, 1060, 1015, 1501, 1023, 1200,
        1340, 1579, 1287, 1062, 1251, 1424, 1516, 1448, 1597, 1575, 1376, 1511,
        1164, 1548, 1555, 1594, 1224, 1470, 1068, 1007, 1104, 1343, 1234, 1152,
        1108, 1079, 1212, 972, 1190, 1271, 1136, 1300, 1003, 1103, 1434, 958,
        1082, 1046, 1326, 1518, 999, 1388, 1472, 1507, 1036, 1316, 1274, 1198,
        1083, 1435, 1387, 1587, 1572, 1290, 1565, 1504, 1127, 1146, 1462, 1268,
        1094, 1520, 1366, 1347, 1483, 1517, 1319, 1092, 1498, 971, 1044, 1034,
        1223, 1346, 1532, 1494, 1123, 1299, 1370, 1298, 1155, 1574, 1240, 1235,
        1264, 1183, 1211, 1586, 1106, 1275, 1027, 1488, 1360, 1490, 1076, 1306,
        1580, 1259, 1592, 1280, 1547, 1114, 1119, 1322, 1446, 1359, 1058, 1011,
        1443, 1307, 1098, 1351, 1598, 1180, 1419, 1508, 995, 1550, 1051, 1194,
        1215, 1247, 1395, 1159, 1531, 1432, 1396, 1276, 1055, 1334, 1272, 1374,
        1355, 1390, 1022, 1571, 967, 1557, 1286, 1228, 975, 1024, 1314, 1158,
        988, 1039, 1182, 955, 1564, 1279, 1544, 1332, 1294, 1308, 1515, 962,
        1420, 1596, 1163, 1047, 1584, 1026, 1436, 1455, 1476, 1403, 1072, 1330,
        1244, 1000, 1510, 1573, 994, 1028, 1549, 1179, 1162, 1552, 1238, 1371,
        1438, 992, 1124, 1367, 1111, 1590, 980, 1242, 1567, 1556, 1054, 1354,
        1539, 1116, 1148, 1004, 1533, 1232, 1339, 1324, 1291, 978, 1048, 1263,
        1582, 1315, 1176, 1248, 1509, 1219, 1407, 1400, 1243, 1172, 1442, 1218,
        1363, 1090, 1067, 1202, 1523, 1187, 1150, 1195, 956, 1452, 1186, 1563,
        1312, 1519, 1427, 1042, 1412, 1595, 1323, 1184, 1086, 1554, 1546, 1246,
        1260, 1479, 1099, 1318, 1368, 1558, 1122, 1384, 1525, 974, 1478, 1118,
        1588, 1418, 1456, 963, 1078, 1408, 1402, 1444, 1142, 983, 1404, 1250,
        1464, 1526, 1207, 1304, 1174, 1019, 1151, 1576, 1358, 1375, 1336, 1192,
        1362, 1102, 1474, 1288, 1296, 1386, 1066, 1056, 970, 1512, 1399, 1416,
        1188, 1070, 1107, 1063, 1295, 1581, 1266, 1012, 1175, 1422, 1134, 979,
        1342, 1154, 1156, 1203, 1168, 1415, 1541, 1132, 1256, 1458, 1482, 1035,
        1196, 1583, 1530, 1310, 1328, 1143, 1100, 1506, 1135, 1451, 1147, 1191,
        1591, 960, 1110, 1414, 1383, 964, 1335, 1231, 1210, 1535, 1394, 1262,
        959, 1214, 1350, 1570, 1084, 1495, 1020, 1071, 1568, 1380, 1144, 1487,
        1222, 1199, 1538, 1160, 1578, 1468]


# Callback for loss history
class LossHistory(keras.callbacks.Callback):
    """
    Keeps track of loss history during training, for each minibatch
    """

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


TRAIN_CAP = None


class DataGenerator(object):
    '''
    Generate minibatches from serialized data.
    '''

    def __init__(self, datadict, shuffle=False, seed=None):
        self.lock = threading.Lock()
        self.data = datadict
        self.shuffle = shuffle
        self.seed = seed
        self.n_songs = len(datadict)
        self.cur_song = 0
        self.sidstr = [sid for sid in datadict]

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        # for python 2.x
        # Keep under lock only the mechainsem which advance the indexing of each batch
        # see # http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            song_idx, self.cur_song = self.cur_song, self.cur_song + 1

        bX, bY = (None, None)
        if song_idx < self.n_songs:
            x_path = self.data[self.sidstr[song_idx]]['X_path']
            y_path = self.data[self.sidstr[song_idx]]['y_path']
            bX = np.memmap(
                x_path,
                dtype='float32',
                mode='r',
                shape=tuple(self.data[self.sidstr[song_idx]]['X_shape'])
            )
            bY = np.memmap(
                y_path,
                dtype='float32',
                mode='r',
                shape=tuple(self.data[self.sidstr[song_idx]]['y_shape'])
            )
            return bX, bY
        else:
            raise StopIteration()
        return bX, bY


def main(
        datadict_train,
        datadict_val,
        datadict_test,
        num_epochs=1,
        batch_size=128,
        learning_rate=1e-4,
        datadir=os.path.abspath('../salami-audio/'),
        salamidir=os.path.abspath('../salami-data-public/'),
        outputdir=os.path.abspath('./bindata/'),
        reg_amount=0.01
):
    """
	Main function
	"""

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

    print("Compiling the model...", )
    model.compile(loss='msle', optimizer=sgd)
    print("Done.")

    # FIT MODEL ###############################################################

    # Callback for model checkpointing
    checkpointer = ModelCheckpoint(
        filepath=os.path.abspath(os.path.join(outputdir, "weights.hdf5")),
        verbose=1,
        save_best_only=True
    )

    history = LossHistory()

    losses = []

    # TODO insert epochs - no just 1

    # Load the training dict
    train = np.load(datadict_train).tolist()

    # Get some validation data
    val = np.load(datadict_val).tolist()
    print(".")
    val_str = [valstr for valstr in val][0]  # Just validate on 1 song
    Xval = np.memmap(
        val[val_str]['X_path'],
        dtype='float32',
        mode='r',
        shape=tuple(val[val_str]['X_shape'])
    )
    yval = np.memmap(
        val[val_str]['y_path'],
        dtype='float32',
        mode='r',
        shape=tuple(val[val_str]['y_shape'])
    )
    n_val = val[val_str]['y_shape'][0]

    for epoch in xrange(num_epochs):
        print("Meta-epoch {0} of {1} ~~~~~~~~~~~~~~~~~~~~~~~~~~").format(epoch + 1, num_epochs)
        # Train on each song in the training set
        for song_str in train:
            print("Training SID " + song_str)
            bX = np.memmap(
                train[song_str]['X_path'],
                dtype='float32',
                mode='r',
                shape=tuple(train[song_str]['X_shape'])
            )
            by = np.memmap(
                train[song_str]['y_path'],
                dtype='float32',
                mode='r',
                shape=tuple(train[song_str]['y_shape'])
            )
            rand_val_idx = np.random.choice(n_val, batch_size * 3)
            hist = model.fit(
                bX, by,
                nb_epoch=1,
                batch_size=batch_size,
                validation_data=(Xval[rand_val_idx], yval[rand_val_idx]),
                shuffle=True,
                verbose=1,
                callbacks=[checkpointer]
            )
            losses.append(hist.history['loss'])

    np.save(
        os.path.abspath(os.path.join(outputdir, 'train_history.npy')),
        np.asarray(losses)
    )

    # SAVE SOME PLOTS
    plt.figure(1)
    plt.plot(losses)
    plt.xlabel('Song')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.savefig(
        os.path.abspath(os.path.join(outputdir, 'loss_history_train.pdf')),
        bbox_inches='tight'
    )
    plt.show()

    # TEST MODEL ###############################################################

    test = np.load(datadict_test).tolist()

    print("Testing model ~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # Test each song in the training set
    for song_str in test:
        print("Testing SID " + song_str)
        bX = np.memmap(
            test[song_str]['X_path'],
            dtype='float32',
            mode='r',
            shape=tuple(test[song_str]['X_shape'])
        )
        # Ground truth:
        by = np.memmap(
            test[song_str]['y_path'],
            dtype='float32',
            mode='r',
            shape=tuple(test[song_str]['y_shape'])
        )
        y_pred = model.predict(bX, batch_size=batch_size, verbose=1)

        plt.figure(3)
        plt.clf()
        plt.plot(y_pred, label="Prediction")
        plt.plot(by, label="Ground truth")
        plt.grid()
        plt.legend()
        plt.title('Test predictions for SID {0}'.format(song_str))
        plt.savefig(
            os.path.abspath(os.path.join(outputdir, 'test' + song_str + '.pdf')),
            bbox_inches='tight'
        )

        plt.show()

        np.savez(
            os.path.abspath(os.path.join(outputdir, 'train_pred_' + song_str + '.npz')),
            y_pred=y_pred,
            y_true=by,
            sid=int(song_str)
        )
    print("All done. Bye.")


if __name__ == "__main__":
    P = argparse.ArgumentParser(
        description='Run a CNN for music structure segmentation.'
    )
    P.add_argument(
        '-t', '--train',
        help='Path to data dictionary for training set.',
        required=True,
    )
    P.add_argument(
        '-v', '--val',
        help='Path to data dictionary for validation set.',
        required=True,
    )
    P.add_argument(
        '-s', '--test',
        help='Path to data dictionary for test set.',
        required=True,
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
        default=sys.maxint
    )

    ARGS = P.parse_args()

    # Start the show
    main(
        datadict_train=ARGS.train,
        datadict_val=ARGS.val,
        datadict_test=ARGS.test,
        num_epochs=int(ARGS.epochs),
        batch_size=int(ARGS.batchsize),
        learning_rate=float(ARGS.learningrate),
        salamidir=os.path.abspath(ARGS.salamidir),
        outputdir=os.path.abspath(ARGS.workingdir),
        reg_amount=float(ARGS.regamount)
    )
