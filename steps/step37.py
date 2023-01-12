# step37에서는 변수값이 스카라가 아닌 벡터일 때를 다룰려고 한다.

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from A_pk import Variable
import numpy as np
import A_pk.functions as F

# 스칼라 일때의 계산
x = Variable(np.array(1.0))
y = F.sin(x)
print(y)

# 행렬의 경우
x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sin(x)
print(y)


# 행렬별 덧셈
x = Variable(np.array([[1,2,3],[4,5,6]]))
c = Variable(np.array([[10,20,30],[40,50,60]]))
y = x + c
print(y)
