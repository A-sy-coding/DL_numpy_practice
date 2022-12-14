# step19에서는 변수 클래스의 사용선 개선을 구현하도록 한다.

import numpy as np
import weakref

class Config:
    enable_backprop = True    

class Variable:
    def __init__(self, data, name=None):
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
        self.name = name

    def cleargrad(self):
        self.grad = None

    def set_creator(self, f):
        self.creator = f
        self.generation = f.generation + 1

    def backward(self, retain_grad = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            funcs.append(f)
            seen_set.add(f)
            funcs.sort(key=lambda x : x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                    
                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # 마지막 grad만 남도록 한다.

    ###### Variable에서 필요한 인스턴스 함수 정의
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    # len 함수 사용
    def __len__(self):
        return len(self.data)

    # print 함수 재정의
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n'+ ' '*9)
        return 'variable(' + p + ')'

    # 연산자 오버로드 구현 (연산자로 계산하기 위해)
    def __mul__(self, other):
        return mul(self, other)

    def __add__(self, other):
        return add(self, other)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs):
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys, )

        outputs = [Variable(as_array(y)) for y in ys]
        
        # 추론시에는 grad를 구할 필요가 없다.
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Sqaure(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = x * 2 * gy
        return gx

def square(x):
    return Square()(x)

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y, )

    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)

class Mul(Function):
    '''
    곱하기의 역전파는 서로 뒤바뀌어서 작동된다.
    '''
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return x1*gy, x0*gy

def mul(x0, x1):
    return Mul()(x0, x1)

import contextlib

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

x = Variable(np.array([[1,2,3],[4,5,6]]))
print('x의 shape 확인 : ',x.shape)
print('x의 ndim 확인 : ', x.ndim)
print('x의 dtype 확인 : ', x.dtype)
print('x의 길이 확인 : ', len(x))

# ㅍVariable class에서 재정의한 print 출력
x1 = Variable(np.array([1,2,3]))
x2 = Variable(np.array(None))
x3 = Variable(np.array([[1,2,3],[4,5,6]]))
print(x1)
print(x2)
print(x3)


##########
# Mul 클래스 사용
a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

# y = add(mul(a,b),c)
y = a * Variable(np.array(2.0)) + c
y.backward()

print('------------------')
print(y)
print(a.grad)
print(b.grad)

x.cleargrad()

with no_grad():
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))

    print(y)
    print(a.grad)
    print(b.grad)

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
y = b * a
print(y)
