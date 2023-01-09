import numpy as np
import weakref

# 다시 시작하기에 앞서 앞에까지 진행했던 사항 복습
class Config:
    enable_backprop = True

class Variable:
    def __init__(self, data, name=None):
        self.data = data
        self.name = name
        self.creator = None
        self.grad = None
        self.generation = 0

    def cleargrad(self):
        self.grad = None

    def set_creator(self, f):
        self.creator = f
        self.generation = f.generation + 1

    def backward(self, retain_grad = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # func 값들을 리스트에 저장
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)


        # 가장 초기에 함수 하나를 저장하고 시작한다.
        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            # 각각의 Variable마다의 grad 저장
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

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'

        p = str(self.data).replace('\n', '\n' + ' '*9)
        return 'variable(' + p + ')'

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(x):
    if not isinstance(x, Variable):
        return Variable(x)
    return x

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys, )

        outputs = [Variable(as_array(y)) for y in ys]

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
    def forawrd(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * 2 * x
        return gx

def sqaure(x):
    return Sqaure()(x)

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)

    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return x1*gy, x0*gy

def mul(x0, x1):
    x1 = as_array(x1) 
    return Mul()(x0, x1)

class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy
        
def neg(x):
    return Neg()(x)


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

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0/ x1 ** 2)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c

    def backward(self, gy):
        x = self.inputs[0].data
        gx = self.c * x ** (self.c-1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)
        
def setup_variable():
    Variable.__add__ = add
    Variable.__radd__= add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow

setup_variable()
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


# step28 에서는 함수 최적화에 대해서 알아보려고 한다.
'''
최적화는 어떤 함수가 주어졌을 때 손실값이 최소화되는 
매개변수값들을 찾는 것이다.
'''

# 여기서는 로젠브록 함수의 최소값을 찾는 연습을 한다.
def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

y = rosenbrock(x0, x1)
y.backward()

print(x0.grad, x1.grad)

x0.cleargrad()
x1.cleargrad()

# 경사 하강법 구현해보기
x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001
iters = 10000

for i in range(iters):
    if i % 50 == 0:
        print('------{}번째 반복 -----'.format(i))
        print(x0, x1)

    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()

    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad



