# step 23에서는 지금까지 만들었던 내용들을 하나의 페키지로 정리하고자 한다.
# step24에서는 복잡한 계산을 수행하는 함수를 구현하려고 한다.

'''
모듈 : 모듈은 파이선 파일이다.
패키지 : 여러 모듈을 묶은 것
라이브러리 : 여러 패키지들을 묶은 것
'''

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import os, sys
print(os.path.dirname(__file__))
print(os.path.join(os.path.dirname(__file__), '..'))

from A_pk import Variable
import numpy as np

x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)

## 여러가지 람수들을 이용해 미분계산 수행

def sphere(x, y):
    ''' z = x^2 + y^2 식'''
    z = x ** 2 + y ** 2
    return z

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
y1 = 1.0
z = sphere(x, y1)
z.backward()
print(x.grad, y.grad)
x.cleargrad()

def matyas(x, y):
    ''' z = 0.26(x**2 + y**2) - 0.48xy '''
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z

z = matyas(x, y)
z.backward()
print(x.grad, y.grad)
