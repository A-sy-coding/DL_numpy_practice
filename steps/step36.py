# step36에서는 역전파때 만들어지는 계산그래프를 고차미분을 제외하고 또 어디에 쓰일수 있는지 확인

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from A_pk import Variable

x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph=True)

gx = x.grad
x.cleargrad()

z = gx ** 3 + y
z.backward(create_graph=True)

print(x.grad)
