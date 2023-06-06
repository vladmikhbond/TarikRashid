import numpy as np

from utils import load_data

X, Y = load_data('.\\mnist_dataset\\mnist_train.csv')
Xt, Yt = load_data('.\\mnist_dataset\\mnist_test.csv')

np.save('X.npy', X)
np.save('Y.npy', Y)
np.save('Xt.npy', Xt)
np.save('Yt.npy', Yt)


