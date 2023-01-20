# step45에서는앞에서 만든 Layer class 를 한꺼번에 관리할 수 있도록 확장하려고 한다.

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 즉, Layer class 안에 Layer 클래스가 담길 수 있도록 구현하고자 한다.

import A_pk.layers as L
import A_pk.functions as F
from A_pk.layers import Layer
from A_pk import Model, Variable
from A_pk.models import MLP
import numpy as np

model = Layer()
model.l1 = L.Linear(5)
model.l2 = L.Linear(3)

def predict(model, x):
    y = model.l1(x)
    y = F.sigmoid(y)
    y = model.l2(y)
    return y

# parameter
for p in model.params():
    print(p)

model.cleargrads()

# 위의 Linear 를 2층 신경망으로 한번에 정의하도록 하는 클래스
class TwoLayerNet(Model):
    ''' Linear class를 2층 신경망'''
    def __init__(self, hidden_size, out_size):
        super().__init__()

        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


#-- TwoLayerNet 확인
x = Variable(np.random.randn(5,10), name='x')
#model = TwoLayerNet(100, 10)
#model.plot(x)

# MLP class 를 사용한 간단한 학습 진행
np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2* np.pi * x) + np.random.rand(100,1)

lr = 0.2
max_iter = 10000

model = MLP((10,20,30,1))

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y_pred, y)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data

    if i%1000 == 0:
        print(loss)



