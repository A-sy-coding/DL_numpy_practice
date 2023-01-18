# step43에서는 affine 변환을 수행하려고 한다.

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2*np.pi*x) + np.random.rand(100,1)
print(y.shape)

#plt.figure(figsize=(15,4))
#plt.plot(x, y, 'o')
#plt.show()

# 위의 데이터는 선형회귀 문제로 풀 수 없기 때문에 신경망을 이용하여 해결 할 수 있다.

# 신경망 추론 구조 
import numpy as np
from A_pk import Variable
import A_pk.functions as F

# dataset
np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2*np.pi*x) + np.random.rand(100,1)

# weight initial
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H,O))
b2 = Variable(np.zeros(O))

# predict function -> 2층 신경망
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

lr = 0.2
iters = 10000

# training
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.cleargrad()

    loss.backward() # 역전파 수행
    
    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    
    # 1000회마다 loss 값 출력
    if i % 1000 == 0:
        print('{} 번째 loss : {}'.format(i, loss))

print('final weight : ', W1.data, b1.data, W2.data, b2.data, loss)

# plot 
result = predict(x)
print(result.data.shape)

#plt.figure(figsize=(15,4))
#plt.plot(x, y , 'o' , label= 'ground_truth') 
#plt.plot(x, result.data, 'ro', label='predict linear')
#plt.legend()
#plt.show()
