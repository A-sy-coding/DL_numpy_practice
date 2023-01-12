# step38에서는 형상 변환 함수를 구현하려고 한다.

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
'''
이전 단계에서는 텐서의 역전파에 대해 다루었다.
여기서는 원소별로 계산하지 않는 함수의 역전파에 대해 알아보려고 한다.
x.data.shape == x.grad.shape이 되도록 구현해야 한다.
'''

import numpy as np
from A_pk import Variable
import A_pk.functions as F

x = np.array([[1,2,3],[4,5,6]])
y = np.reshape(x, (6,))
print(y)

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)

print(x)
print(x.grad)
print(y)
print(y.grad)

# reshape를 수행할 때 tuple, list, 인자 그대로 받는 것을 반영하도록 한다. -> Variable class 변형
x = Variable(np.random.randn(1,2,3))
y = x.reshape((2,3))
y1 = x.reshape([2,3])
y2 = x.reshape(2,3)

# 전치 구현하기
x = np.array([[1,2,3],[4,5,6]])
y = np.transpose(x)
print(x)
print(y)

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.transpose(x)
y.backward(retain_grad=True)
# transpose 했을 때 역전파와 순전파의 shape들이 유지되고 있음을 확인할 수 있다.
print(x)
print(x.grad)
print(y)
print(y.grad)

# Variable 자체에서도 transpose 를 사용가능하게 한다.
x = Variable(np.random.rand(2,3)) # 2*3 matrix
y = x.T
y1 = x.transpose()

print(x.shape)
print(y.shape)
print(y1.shape)

# np의 transpose 범용 활용
A, B, C, D = 1,2,3,4
x = np.random.rand(A,B,C,D)
print(x.shape)
y = x.transpose(1,0,3,2) # index의 번호로 각 shape을 변경
print(y.shape)
