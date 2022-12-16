# step21_22에서는 연산자 오버로드에 관해 구현하려고 한다.
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

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            funcs.append(f)
            seen_set.add(f)
            funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [y().grad for y in f.outputs]
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
                    y().grad = None

    # 필요한 기능 추가
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    # len
    def __len__(self):
        return len(self.data)

    # print
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'

        p = str(self.data).replace('\n', '\n' + ' '*9)
        return 'variable(' + p + ')'

    # overload
    def __mul__(self, other):
        return mul(self, other)

    def __add__(self, other):
        return add(self, other)

    

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [input.data for input in inputs]
        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys, )

        outputs = [Variable(as_array(y)) for y in ys]

        # 추론할 때는 backward 필요 없음
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

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)

    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return x1*gy, x0*gy

def mul(x0, x1):
    return Mul()(x0, x1)

class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)

Variable.__neg__ = neg

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

Variable.__sub__ = sub
Variable.__rsub__ = rsub

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)
    
Variable.__truediv__  = div
Variable.__rtruediv__ = rdiv

class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.intputs[0].data
        gx = self.c * x ** (c-1) * gy
        return gx

def pow(x,c):
    return Pow(c)(x)

Variable.__pow__ = pow

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

x = Variable(np.array(3.0))
x1 = Variable(np.array(2.0))
y = add(square(x), square(x1))
y.backward()
print(y.data)
print(x.grad)
print(y.shape)
print(y.ndim)
print(y.dtype)

z = Variable(np.array([[1,2,3],[4,5,6]]))
print(len(z))

a = x + x1
b = x * x1
print(a)
print(b)

# 현재는 Variable 끼리만 계산이 가능하다.
# 인스턴스와 계산이 가능하도록 만들어야 한다,
'''
ndarray가 오면, Variable이 되도록 변경해준다.
'''
x = Variable(np.array(2.0))
y = np.array(2.0) + np.array(3.0)
y1 = x + np.array(3.0)
print(y)
print(y1)

x = Variable(np.array(2.0))
y = -x
print(y)

x = Variable(np.array(2.0))
y1 = 3.0 - x
y2 = x - 3.0
print(y1)
print(y2)

x = Variable(np.array(4.0))
y1 = 2.0 / x
y2 = x / 2.0
print(y1)
print(y2)

x = Variable(np.array(2.0))
y = x ** 4
print(y)
