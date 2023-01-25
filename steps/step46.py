# step46에서는 매개변수 갱신 클래스를 정의하고 SGD 경사하강법 수행

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from A_pk import Variable
from A_pk import optimizers
import A_pk.functions as F
from A_pk.models import MLP

np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2*np.pi*x) + np.random.rand(100,1)

lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP((hidden_size, 1)) # 인자값으로 output size들이 들어가게 된다.(층중 신경망)
# optimizer = optimizers.SGD(lr)
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model)  # MLP model를 target으로 설정

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()
    
    # param update
    optimizer.update()
    if i % 1000 == 0:
        print(loss)
