# step50에서는 iter 관련 함수를 사용하여 DataLaoder를 구현하려고 한다.

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class MyIterator:
    def __init__(self, max_cnt):
        self.max_cnt = max_cnt
        self.cnt = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt == self.max_cnt:
            raise StopIteration

        self.cnt += 1
        return self.cnt

obj = MyIterator(5)
for x in obj:
    print(x)

from A_pk.datasets import Spiral
from A_pk import DataLoader

batch_size = 10
max_epoch = 1

train_set = Spiral(train=True)
test_set = Spiral(train=False)

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

for epoch in range(max_epoch):
    for x, t in train_loader:
        print(x.shape, t.shape)
        break

# accuracy 확인
import numpy as np
import A_pk.functions as F
from A_pk import optimizers


y = np.array([[0.2,0.8,0], [0.1,0.9,0], [0.8,0.1,0.1]])
t = np.array([1,2,0])
acc = F.accuracy(y, t)
print(acc)

#-- DataLoadder와 accuracy를 활용하여 데이터셋 학습 진행
import A_pk
from A_pk.models import MLP


max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = A_pk.datasets.Spiral(train=True)
test_set = A_pk.datasets.Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('epoch : {}'.format(epoch+1))
    print('train loss : {:.4f}'.format(sum_loss / len(train_set)))
    print('accuracy : {:.4f}'.format(sum_acc / len(train_set)))

    sum_loss, sum_acc = 0, 0
    with A_pk.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('test loss : {:.4f}, accuracy: {:.4f}'.format(sum_loss/len(test_set), sum_acc / len(test_set)))
