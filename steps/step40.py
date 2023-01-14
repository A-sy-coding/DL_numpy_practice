if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# step40에서는 broadcast를 구현해보려고 한다.
import numpy as np

x = np.array([1,2,3])
y = np.broadcast_to(x, (2,3))
print(y)

from A_pk.utils import sum_to

x = np.array([[1,2,3],[4,5,6]])
print(x.ndim)
print(x)
y = sum_to(x, (1,3))
print(y)

y = sum_to(x, (2,1))
print(y)

# broadcast check
x0 = np.array([1,2,3])
x1 = np.array([10])
y  = x0 + x1
print(y)

'''
위의 브로드캐스트 계산이 Variable 에서 이뤄지려면 
Add class에서 덧셈을 할 때 broadcast_to를 이용하여 역전파는 sum_to가 되도록 한다.
'''

from A_pk import Variable

x0 = Variable(np.array([1,2,3]))
x1 = Variable(np.array([10]))
y = x0 + x1
print(y)

y.backward()
print(x1)
print(x1.grad)

