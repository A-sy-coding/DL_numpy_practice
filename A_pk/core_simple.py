# core_simple 에서는 Config, Variable, Functioin들을 정리하려고 한다.

import numpy as np
import weakref
import A_pk

class Config:
    enable_backprop = True

class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.creator = None
        self.grad = None
        self.name = name
        self.generation = 0

    def set_creator(self, f):
        self.creator = f
        self.generation = f.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad = False, create_graph=False):
        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            # 고차미분을 수행하기 위해서 grad를 Variable class로 변경
            self.grad = Variable(np.ones_like(self.data))            

        funcs = []
        seen_set = set()

        def add_func(f):
            funcs.append(f)
            seen_set.add(f)
            funcs.sort(key=lambda x : x.generation) 
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [y().grad for y in f.outputs]

            # 이제부터 function의 backward도 계산 그래프를 생성해야 한다.
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)

                if not isinstance(gxs, tuple):
                    gxs = (gxs, )

                for gx, x in zip(gxs, f.inputs):
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

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n'+' '*9)
        return 'variable(' + p + ')'

    def reshape(self, *shape):
        print('shape 확인 : ',shape)
        '''
        Variable type에서 reshape를 사용시 tuple, list, 인자값 각각을 받기 위한 처리
        '''
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        return A_pk.functions.reshape(self, shape) 
   
    def transpose(self):
        return A_pk.functions.transpose(self)

    @property
    def T(self):
        return A_pk.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return A_pk.functions.sum(self, axis, keepdims)

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
        # numpy 도는 실수값들이 연산가능하도록 구현
        inputs = [as_variable(input) for input in inputs]

        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        
        if not isinstance(ys, tuple):
            ys = (ys, )

        outputs = [Variable(as_array(y)) for y in ys]

        # 추론시
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
        # x = self.inputs[0].data
        x = self.inputs
        gx = gy * 2 * x
        return gx

def sqaure(x):
    return Sqaure()(x)

class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape # broadcast 수행 준비
        y = x0 + x1
        return (y,)

    def backward(self, gy):
        gx0, gx1 = gy, gy

        # shape가 같지 않으면, broadcast가 수행되도록 한다.
        if self.x0_shape != self.x1_shape:
            gx0 = A_pk.functions.sum_to(gx0, self.x0_shape) 
            gx1 = A_pk.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1 

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
       # x0, x1 = self.inputs[0].data, self.inputs[1].data
        '''
        inputs는 Variable 클래스로 구성되어 있는 list이다.
        역전파를 수행할 때 Variable class이어야지 계산그래프가 생성된다.
        따라서, Variable안의 data를 가져오는 것이 아니라 Varaible 그 자체를 가져오도록 
        코드를 변경한다.
        
        또한, return에서 Variable끼리의 계산은 기존에 오버라이딩 한 계산식에 의해 작동된다.
        '''
        x0, x1 = self.inputs

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
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        x0, x1 = self.inputs
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
        # x = self.inputs[0].data
        x, = self.inputs
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
