# step39에서는 sum을 할 때 벡터값도 받을 수 있게 구현하려고 한다.

'''
벡터값을 받았을 때 각 순전파값의 shape과 역전파 값의 shape이 유지되어야 한다.
'''

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from A_pk import Variable
import A_pk.functions as F

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum(x)
print(y)

y.backward()
print(x)
print(x.grad)

# axis 와 keepdims 적용
x = np.array([[1,2,3],[4,5,6]])
y = np.sum(x, axis=0)
y1 = np.sum(x, axis=1)
print(y)
print(y1)
print(x.shape)
print(y.shape)
print(y1.shape)

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum(x, axis=0)
y.backward()
print(y)
print(x.grad)

x = Variable(np.random.randn(2,3,4,5))
y= x.sum(keepdims=True)
print(x.shape)
print(y.shape)
print(y)
