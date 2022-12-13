# step17은 메모리 관리에 대해서 구현하려고 한다.
import numpy as np
import weakref

class Config:
    enable_backprop = True

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, f):
        self.creator = f
        self.generation = f.generation + 1
        self.generation = 0
    def cleargrad(self):
        self.grad = None


    def backward(self, retain_grad = False):
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
                    y().grad = None # 참조 제거

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

        # 추론을 할 때는 grad를 활용할 필요가 없다.
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
        return x**2

    def backward(self, gy): 
        for input in self.inputs:
            print('--- input 데이터 확인 : ',input.data)
        print('------------')
        x = self.inputs[0].data
        gx = 2 * x * gy

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


x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)
    
#################
# 참조 카운트 방시의 메모리 관리
'''
모든 객체는 참조 카운트가 0인 상태로 생성
참조가 끊길 때마다 참조 카운트 감소
'''

#################
# GC 메모리 관리 방식
'''
메모리가 부족해지는 시점에 파이선 인텊리터에 의해
자동으로 호츨되게 된다.

다만, 참조 카운트 방식의 메모리 관리보다 메모리를 많이 차지하기 때문에
딥러닝에서는 순환참조를 만들지 않는 것이 중요하다.
'''

##################
# weakref 함수를 사용하여 약한 참조 구성
'''
약한 참조는 다른 참조를 참조하되 참조 카운트는 증가시키지 않는 기능이다.
'''
import weakref

a = np.array([1,2,3])
b = weakref.ref(a)

print(b)
print(b())

a = None
print(b)

##############
# 역전파용 메모리 절약 방법
x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))

t = add(x0, x1)
y = add(x0, t)
y.backward()

print(y.grad, t.grad)
print(x0.grad, x1.grad)

'''
위의 경우는 모든 변수값들이 미분값을 가지고 있다.
하지만, 우리가 원하는 미분값은 주로 x0, x1이다.
따라서, 나머지 미분값들은 저장되지 않도록 하여 메모리양을 줄인다.
'''


###########
# 슌전파 역전파 모드 변환 확인
Config.enable_backprop = True
x = Variable(np.ones((100,100,100)))
y = square(square(square(x)))
y.backward()
print('x.grad 값 확인 : ',x.grad)

Config.enable_backprop = False
x = Variable(np.ones((100,100,100)))
y = square(square(square(x)))
print('y값 확인 : ', y.data)

###########
# 구현 모드 전환 함수 추가 수정
import contextlib

@contextlib.contextmanager
def using_config(name, value):
    '''
    기존 값을 저장해놓았다가
    값을 변경한 후 코드를 실행하고
    코드가 끝나면 다시 원상태로 복구하도록 한다.
    '''
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

with no_grad():
    x = Variable(np.array(2.0))
    y = square(x) 
    print('y 값 확인 : ', y.data)

