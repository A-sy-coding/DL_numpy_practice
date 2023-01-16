# step43에서는 벡터의 내적과 행렬의 곱에 대해서 다루도록 한다.

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# 벡터의 내적
a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.dot(a,b)
print(c)

# 행렬의 곱
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
print(np.dot(a,b))

# Variblae matmul 수행
from A_pk import Variable
import A_pk.functions as F

x = Variable(np.random.randn(2,3))
W = Variable(np.random.randn(3,4))

y = F.matmul(x, W)
print(y)
print(y.shape)

y.backward()
print(x.grad.shape)
print(W.grad.shape)
