# step49에서는 Dataset class를 이용하여 데이터를 받아오려고 한다.

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import A_pk.datasets

train_set = A_pk.datasets.Spiral(train=True)
print(train_set[0])
print(len(train_set))

# data concat
batch_index = [0,1,2]
batch = [train_set[i] for i in batch_index]
print(batch)

'''
A_pk의 신경망의 input으로 들어가기 위해서는 ndarray형태로 들어가야 한다.
'''
import numpy as np
from A_pk.models import MLP
import A_pk.functions as F
import math
from A_pk import optimizers

x = np.array([example[0] for example in batch])
t = np.array([example[1] for example in batch])

print(x.shape)
print(t.shape)
print(x)

#-- Spiral class 를 사용하여 학습 진행
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = A_pk.datasets.Spiral()
model = MLP((hidden_size, 10)) 
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

#for epoch in range(max_epoch):
#    index = np.random.permutation(data_size)
#    sum_loss = 0
#
#    for i in range(max_iter):
#        # batch size
#        batch_index = index[i*batch_size : (i+1)* batch_size]
#        batch = [train_set[i] for i in batch_index]
#        batch_x = np.array([example[0] for example in batch])
#        batch_t = np.array([example[1] for example in batch])
#
#        y = model(batch_x)
#        loss = F.softmax_cross_entropy(y, batch_t)
#        model.cleargrads()
#        loss.backward()
#        optimizer.update()
#
#        sum_loss += float(loss.data) * len(batch_t)
#
#    avg_loss = sum_loss / data_size
#    print('epoch {}, loss {:.2f}'.format(epoch+1, avg_loss))


#-- 전처리 함수 확인하기
def f(x):
    y = x / 2.0
    return y

origin = A_pk.datasets.Spiral()
sample1 = A_pk.datasets.Spiral(transform=f)
print(origin[0])
print(sample1[0])

from A_pk import transforms
f = transforms.Normalize(mean=0.0, std=2.0)
sample2 = A_pk.datasets.Spiral(transform=f)
print(sample2[0])


f = transforms.Compose([transforms.Normalize(mean=0.0, std=2.0),
                        transforms.AsType(np.float64)])
sample3 = A_pk.datasets.Spiral(transform=f)
print(sample3[0])

