import numpy as np
import matplotlib.pyplot as plt

def load_data(path):    
    # Завантаження CSV-файлу у масив
    data = np.genfromtxt(path, delimiter=',')
    n = len(data)

    # Зведення входів до діапазону [0.01, 1.0]
    X = np.delete(data, 0, axis=1)
    X = X / 255.0 * 0.99 + 0.01
    
    # Зведення виходів
    Y = np.zeros((n, 10)) + 0.01
    for i in range(n):
       Y[i, int(data[i, 0])] = 0.99    
    return X, Y


def show_digit(arr):
    arr = (arr - 0.01) * 255
    img_data = arr.reshape((28,28))
    plt.imshow(img_data, cmap='Greys', interpolation='None')
    plt.show(block=True)


def show_err_history(arr):
    plt.plot(arr)
    plt.show(block=True)

