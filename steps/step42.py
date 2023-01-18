# step42에서는 지금까지 쌓아올린 프레임워크를 이용하여 선형회귀를 구현하려고 한다.

if '__file__' in  globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from A_pk import Variable
import A_pk.functions as F

# toy dataset setting
np.random.seed(0)
x = np.random.rand(100,1)
#print(x.shape)

y = 5 + 2 * x + np.random.rand(100,1)

import matplotlib.pyplot as plt
# plt.figure(figsize=(14,4))
#x1 = np.linspace(0,1,100)
#print(x1.shape)
#plt.scatter(x, y)
#plt.show()

'''
선형회귀는 y = Wx + b 꼴의 형태를 가지고 있다.
이때, 손실값으로는 MSE를 사용하고, 손실값을 최소화하는 W, b를 찾도록 한다.
'''

# linear regression 구현
x, y = Variable(x), Variable(y)
x_np, y_np = x.data, y.data

W = Variable(np.zeros((1,1)))
b = Variable(np.zeros(1))
print(W.shape)
print(W)
print(b.shape)

def predict(x):
    y = F.matmul(x, W) + b
    return y

# Loss
def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

# learning rate and iter_num
lr = 0.1
iters = 100

y_store = []
for i in range(iters):
    y_pred = predict(x)
    y_store.append(y_pred.data[0][0])
    loss = F.mean_squared_error(y_pred, y)
    
    W.cleargrad()
    b.cleargrad()
    loss.cleargrad()
    loss.backward()

    # 경사 하강법
    W.data -= W.grad.data * lr
    b.data -= b.grad.data * lr
    print(W, b, loss)

print('-----')
print(x_np.shape)

result = np.matmul(x_np,W.data) + b.data
print(result.shape)

# linear plot 그리기
#plt.figure(figsize=(15,4))
#plt.plot(x_np, result, 'r-', label='predict linear')
#plt.plot(x_np, y_np, 'bo', label='ground truth')
#plt.legend()
#plt.show()

# mean squared error 함수 수정
'''
인자로 들어가는 x0, x1이 Variable이기 때문에 계산 그래프가 생성되게 된다.
MSE의 계산그래프를 그려보면, 중간에 이름없는 변수가 존재하게 되고, 해당 변수들은 메모리를
계속 차지하게 된다.
따라서, 해당 변수들이 메모리를 차지하지 않게 하기 위해서는 해당 계산을 ndarray 인스턴스로 변경하면 된다.
'''

