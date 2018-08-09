import numpy as np

A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))
print(A.shape)

B = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
print(B)
print(np.ndim(B))
print(B.shape)

C = np.array([[[1, 2, 3], [3, 4, 5], [5, 6, 7]],[[1, 2, 3], [3, 4, 5], [5, 6, 7]]])
print(C)
print(np.ndim(C))
print(C.shape)


D = np.array([[1, 2], [3, 4]])
E = np.array([[5, 6], [7, 8]])
# print(np.dot(D, E))
# エラー

F = np.array([[1, 2], [3, 4], [5, 6]])
G = np.array([7, 8, 9])
print(np.dot(G, F))


# ニューラルネットワークの実装

X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
print(np.dot(X, W))

# 行列内積 -> 活性化 -> 行列内積 -> 活性化
