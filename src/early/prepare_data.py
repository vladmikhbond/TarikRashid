import sys 
sys.path.append('c:\\Git\\PY\\TarikRashid\\src\\lib')
sys.path.append(r'c:\users\volodymyr.bondariev\appdata\roaming\python\python311\site-packages')

import numpy as np
import time

from neural_network import neural_network
from utils import show_digit, show_err_history

# Завантаження CSV-файл і розділення на X та Y ---------
path = '.\\early_dataset\\Data.csv'
X = np.genfromtxt(path, delimiter=',', encoding='utf-8', skip_header=1, 
                       usecols = (0, 3,4,5,6,7, 9,10,11,12,13,14,15,16), missing_values='nan', filling_values=0)
Y = np.genfromtxt(path, delimiter=',', encoding='utf-8', skip_header=1, usecols = (8,))
assert(len(X) == len(Y))

# Нормалізація входів [0, 1] ----------------------------
X_min = np.min(X, axis=0)
X_delta = np.max(X, axis=0) - X_min
X = (X - X_min) / X_delta
X = X * 0.98 + 0.01

# [0, 1] - не здав, [1, 0] - здав
Y = [ [0.01, 0.99] if y < 60 else [0.99, 0.01] for y in Y]

# Поділення на навчальний і тестовий сети ------------------
Q = 200
Xt, Yt = X[Q:], Y[Q:]
X, Y = X[:Q], Y[:Q]


# Навчання НМ ---------------------------------------------
start =  time.time() # -----
err_history = []

net = neural_network(len(X[0]), 100, 2, 0.2)
N = len(X)
epochs = 3
for еpoch in range(epochs):
   for i in range(N):
      err = net.train(X[i], Y[i])      
      err_history.append(err)
        
print("Net Learned", time.time() - start) # -----
# show_err_history(err_history)


# Перевірка нейронної мережі не тестовому сеті ------------------------

ok = 0
fails = []
ys_hat = []
ys = []
for i in range(len(Xt)):
   y_hat = np.argmax(net.query(Xt[i]))
   y = np.argmax(Yt[i])
   if y == y_hat: 
      ok += 1
   else:
      fails.append((i, y, y_hat))
   ys.append(y)
   ys_hat.append(y_hat)
   

print("Success", ok / len(Xt))
print(fails[0:10])

print("avg wih, who:", np.abs(net.wih).mean(), np.abs(net.who).mean())
print("max wih, who:", np.abs(net.wih).max(), np.abs(net.who).max())

# metrics ------------------------------------------------------

from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import precision_score

prec_score = precision_score(ys, ys_hat)
reca_score = recall_score(ys, ys_hat)
f_score = f1_score(ys, ys_hat)

print((prec_score, reca_score, f_score))



