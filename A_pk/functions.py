from A_pk.core_simple import Variable, Function, as_array, as_variable
import numpy as np
import math

# Sin 함수 구현
#class Sin(Function):
#    def forward(self, x):
#        y = np.sin(x)
#        return y
#
#    def backward(self, gy):
#        x = self.inputs[0].data
#        gx = gy * np.cos(x)
#        return gx
#
#def sin(x):
#    return Sin()(x)

def my_sin(x, threshold=0.001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x **(2*i + 1)
        y = y + t

        if abs(t.data) < threshold:
            break

    return y
    
# Sin 함수 구현 -> 고차 미분이 가능하게 구현
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        '''
        backward method안의 계산 값들은 모두 Variable type이어야 한다.
        그래야지 계산 그래프가 생성된다.
        따라서, gx를 구할 때 cos도 Variable 타입이 되어야 한다.
        '''
        print('Sin inputs type 확인 : ', type(self.inputs))
        x, = self.inputs
        gx = gy * cos(x) # 이때 cos(x) 함수는 Variable 타입이어야 한다.
        return gx

def sin(x):
        return Sin()(x)

# Cos 함수 구현
class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x) # 여기서도 Variable 끼리의 계산으로 만들어야 한다.
        return gx

def cos(x):
    return Cos()(x)

# Tanh 함수 구현
class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx

def tanh(x):
    return Tanh()(x)

