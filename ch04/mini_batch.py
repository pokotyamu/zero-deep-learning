import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
batch_mask = np.random.choice(x_train.shape[0], 10)

print(x_train.shape)
print(t_train.shape)
print(batch_mask)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y + 1e-7)) / batch_size

y = np.array([
    [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
    [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.7, 0.0, 0.0],
    [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
    [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
    [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]])
batch = np.arange(5)
"""one-hot だと 0/1 だから、正解ラベル以外は掛けた値が0なので、正解ラベルだけ抽出すればよい"""
t = np.array([2,7,0,9,4])
