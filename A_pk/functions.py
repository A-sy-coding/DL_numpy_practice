from A_pk.core_simple import Variable, Function, as_array, as_variable
import numpy as np
import math
from A_pk import utils


# Log 함수 구현
class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy /  x
        return gx

def log(x):
    return Log()(x)

# Exp 함수 구현
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        return gx

def exp(x):
    return Exp()(x)

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

# mean_square_error function
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        ''' ndarray 계산으로 memory 절약 가능'''
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

# 간단한 linear 신경망 함수
def linear_sample(x, W, b=None):
    '''
    순전파를 수행할 때에는 x, W, b, t 모두 필요하지만,
    역전파를 수행할 때에는 b가 필요없게 된다. (덧셈의 역전파는 그대로 흘러들어가는 것이기 때문)
    '''
    t = matmul(x,W)
    if b is None:
        return t

    y = t + b
    t.data = None
    return y

# Linear Class 구현
class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is None:
            return y

        y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape) # 형상 유지를 위함
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)

# simple Sigmoid 활성화 함수 구현
def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x)) # Variable끼리의 계산
    return y

# Sigmoid 구현
class Sigmoid(Function):
    def forward(self, x):
        y = np.tanh(x + 0.5) * 0.5 + 0.5 # 더 나은 실행 방법
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1-y)
        return gx

def sigmoid(x):
    return Sigmoid()(x)

#-- GetItem 구현 -> slice 연산 구현
class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs  # tuple이기 때문에 ,를 붙혀주어야 한다.
        f = GetItemGrad(self.slices, x.shape) # matrix shape를 유지해야 한다.
        return f(gy)

def get_item(x, slices):
    return GetItem(slices)(x)

class GetItemGrad(Function):
    ''' GetItem의 역전파를 수행해주는 클래스'''
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy): 
        gx = np.zeros(self.in_shape) 
        np.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)
    
#-- Softmax 구현
class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy  # exp의 역전파
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx

def softmax(x, axis=1):
    return Softmax(axis)(x)

# softmax_cross_entropy simple function 구현
def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax(x)
    p = clip(p, 1e-15, 1.0) # log0 방지
    log_p = log(p)
    # print('log_p : ', log_p)
    tlog_p = log_p[np.arange(N), t.data]
    # print('tlog_p : ', tlog_p)
    y = -1 * sum(tlog_p) / N
    return y

# softmax_cross_entropy function 구현
class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1) # softmax를 수행하는 함수
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y-t_onehot) * gy
        return y

def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x,t)

#-- Clip class 구현
class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx

def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)

#-- accuracy
def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))

#-- ReLU function
class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx

def relu(x):
    return ReLU()(x)
