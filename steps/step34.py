if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from A_pk import Variable
import A_pk.functions as F
import numpy as np

# Variable끼리 계산 되는지 임시 확인
a = Variable(np.array(2.0))
b = Variable(np.array(3.0))
c = a * b
print(c)

# Sin 함수의 고차미분 수행해보기
x = Variable(np.array(1.0))
y = F.sin(x)
y.backward(create_graph=True)

print(x.grad)

for i in range(3):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    print('{}번째 미분값 : {}'.format(i+1, x.grad))

# 위의 고차 미분을 그래프로 그려보기
import matplotlib.pyplot as plt

x = Variable(np.linspace(-7,7, 200))
y = F.sin(x)
y.backward(create_graph=True)

#print('y의 type 확인 : ', type(y))
#logs = [y.data]
#
#for i in range(3):
#    logs.append(x.grad.data)
#    gx = x.grad
#    x.cleargrad()
#    gx.backward(create_graph=True)
#
#labels = ["y=sin(x)", "y`", "y``", "y```"]
#for i, v in enumerate(logs):
#    plt.plot(x.data, logs[i], label=labels[i])
#plt.legend(loc='lower right')
#plt.show()
