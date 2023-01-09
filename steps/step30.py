# step30에서는 고차미분도 자동으로 계산 가능하도록 구현하려고 한다.
# 고차 미분의 가장 중요한 idea는Variable 클래스의 grad이다.
# 즉, grad를 Variable class로 변경시키는 것이 키포인트이다.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from A_pk import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))

# 뉴턴 방법을 활용하여 최적화 수행
iters = 10

y = f(x)
y.backward(create_graph = True)
print(x.grad)

gx = x.grad
x.cleargrad()
gx.backward()
print(x.grad)

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    # 갱신
    x.data -= gx.data / gx2.data
