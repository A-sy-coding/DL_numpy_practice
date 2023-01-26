# step47에서는 softmax와 cross-entropy를 구현해보려고 한다.

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from A_pk import Variable, as_variable
import A_pk.functions as F

# array를 slice 해주는 함수 구현
x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.get_item(x, 1)
print(x)
print(y)

'''
slice로 인한 계산의 역전파는 다차원 배열의 데이터 일부를 수정하지 않고 전달하는데에 있다.
'''

x = Variable(np.array([[1,2,3],[4,5,6]]))
indices = np.array([0,0,1])
y = F.get_item(x, indices)
print(x)
print(y)

# get_item을 특수 메서드로 지정
#Variable.__getitem__ = F.get_item
y = x[1]
print(y)

#-- MLP 사용
from A_pk.models import MLP

# softmax function
def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y/sum_y


model = MLP((10,3))

x = np.array([[0.2,-0.4]])
#x = np.array([0.2,-0.4])
y = model(x)
print(y)
p = softmax1d(y)
print(p)

x = np.array([[0.2,-0.4],[0.3,0.5],[1.3,-3.2],[2.1,0.3]])
print(x.shape)
y = model(x)
print(y.shape)
print(y)

p = F.softmax(y)
print('p : ',p)

p.backward()

# softmax_cross_entropy 
x = np.array([[0.2,-0.4], [0.3,0.5], [1.3,-3.2], [2.1,0.3]])
t = np.array([2,0,1,0]) # 3개의 범주형으로 출력
print('t : ', t)

y = model(x)
print('y :' ,y)

# loss = F.softmax_cross_entropy_simple(y, t)
loss = F.softmax_cross_entropy(y, t)

print('loss : ', loss)

loss.backward()
