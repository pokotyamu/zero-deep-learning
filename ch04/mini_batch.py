import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
batch_mask = np.random.choice(x_train.shape[0], 10)

print(x_train.shape)
print(t_train.shape)
print(batch_mask)
