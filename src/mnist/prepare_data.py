import numpy as np

from utils import load_data

X, Y = load_data('.\\mnist_dataset\\mnist_train_100.csv')
Xt, Yt = load_data('.\\mnist_dataset\\mnist_test_10.csv')

np.save('.\\mnist_dataset\\X.npy', X)
np.save('.\\mnist_dataset\\Y.npy', Y)
np.save('.\\mnist_dataset\\Xt.npy', Xt)
np.save('.\\mnist_dataset\\Yt.npy', Yt)


