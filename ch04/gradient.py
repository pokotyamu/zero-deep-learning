import numpy as np
import matplotlib.pylab as plt

def function2(x):
    return x[0]**2 + x[1]**2

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # x と同じ形状の配列を生成

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

def gradient_discent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

print(gradient_discent(function2, np.array([-3.0, 4.0])))
print(gradient_discent(function2, np.array([-3.0, 4.0]), lr=0.1))
print(gradient_discent(function2, np.array([-3.0, 4.0]), lr=10))
print(gradient_discent(function2, np.array([-3.0, 4.0]), lr=1e-10))
"""
[-0.39785867  0.53047822]
[ -6.11110793e-10   8.14814391e-10]
[ -2.58983747e+13  -1.29524862e+12]
[-2.99999994  3.99999992]
なるべく少ない計算回数で収束させたい。
"""
