import numpy as np
import time

from neural_network import neural_network
from utils import load_data, show_digit, show_err_history

X = np.load('X.npy')
Y = np.load('Y.npy')
Xt = np.load('Xt.npy')
Yt = np.load('Yt.npy')

start = time.time() ######

# Навчання НМ
err_history = []

nn = neural_network(28*28, 100, 10, 0.1)
N = len(X)
epochs = 5
for е in range(epochs):
   for i in range(N):
      err = nn.train(X[i], Y[i])
      if i % 500 == 0:
         err_history.append(err)
        
print("Net Learned", time.time() - start) ######
show_err_history(err_history)


# Перевірка нейронної мережі не тестовому сеті

ok = 0
fails = []
for i in range(len(Xt)):
   y_hat = np.argmax(nn.query(Xt[i]))
   y = np.argmax(Yt[i])
   if y == y_hat: 
      ok += 1
   else:
      fails.append((i, y, y_hat))
print("Success", ok / len(Xt))
print(fails[0:10])

print("avg wih, who:", np.abs(nn.wih).mean(), np.abs(nn.who).mean())
print("max wih, who:", np.abs(nn.wih).max(), np.abs(nn.who).max())

i = None
while(True):
   i = int(input())
   show_digit(Xt[i])

