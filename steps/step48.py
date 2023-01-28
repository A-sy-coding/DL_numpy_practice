# step48에서는 다중 클래스 분류를 수행하려고 한다.

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import A_pk.datasets

x, t = A_pk.datasets.get_spiral(train=True)
print(x[10], t[10])
print(x[110], t[110])

# 학습 시작
import math
import numpy as np
from A_pk import optimizers
import A_pk.functions as F
from A_pk.models import MLP

# param
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

x, t = A_pk.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i+1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        
        # backpropogation
        loss.backward()
        
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)

        # print('loss_data : ', loss.data)
        # print('len batch_t : ', len(batch_t))
        

    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch+1, avg_loss))
