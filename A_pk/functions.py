from A_pk.core_simple import Variable, Function, as_array, as_variable
import numpy as np
import math
from A_pk import utils

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


# reshape 클래스 만들기
class Reshape(Function):
    '''
    x.data.shape == x.grad.shape가 되게 구현해야 한다.
    '''
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape  # 원래 x의 shape를 저장한다.
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        '''
        gy가 Variable 타입이므로 reshape를 할 때에도 Variable 전용 reshape를 사용해야 한다.
        '''
        return reshape(gy, self.x_shape)

# Variable reshape 함수 구현
def reshape(x, shape):
    if x.shape == shape:
        as_variable(x)
    return Reshape(shape)(x)

# transpose 클래스 구현
class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y

    def backward(self, gy):
        gx = transpose(gy) # gy가 Variable이므로 transpose라는 함수를 사용하여 계산한다.
        return gx

def transpose(x):
    return Transpose()(x)

# Broadcast 클래스 구현
class BroadcastTo(Function):
    ''' 값을 shape의 크기만큼 복사해주는 역할을 수행한다.'''
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape) # 크기 변경 후 값 복사
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

# sum_to 클래스 구현
class SumTo(Function):
    '''
    BroadcastTo class의 역전파는 입력값과 형상이 같아지도록 기울기의 합을 구하게 된다.
    '''
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

# Sum 클래스 구현
class Sum(Function):
    '''
    shape이 유지되도록 한다.
    axis / keepdims 기능을 추가시킨다.
    '''
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        ''' gy값에 맞게 형식을 복사하고 형태를 유지하도록 한다.'''
        gx = broadcast_to(gy, self.x_shape)
        return gx 

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

# 행렬의 곱 클래스 구현
class Matmul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

def matmul(x, W):
    return Matmul()(x, W)
