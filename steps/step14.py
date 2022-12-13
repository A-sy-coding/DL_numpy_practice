# step14에서는 같은 변수를 사용 가능하도록 한다.
import numpy as np

class Variable:
    def __init__(self, data):
        self.data =  data
        self.grad = None 
        self.creator = None

    def set_creator(self, f):
        self.creator = f
    
    # 미분값을 초기화 해주는 함수 설정
    def cleargrad(self):
        self.grad = None

    # 자동 미분 구현
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        print('funcs 확인 : ', funcs)
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                print('---- x.grad 확인 : ', x.grad)
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
            
                if x.creator is not None:
                    funcs.append(x.creator)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs):
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        
        # ys가 ndarray가 아닌 np.float이 될 수도 있으므로 tuple로 변경
        if not isinstance(ys, tuple):
            ys = (ys, )

        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            print('--- output 값 ㅎ확인 : ', output)
            output.set_creator(self)
        
        self.inputs = inputs
        self.outputs = outputs

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

x = Variable(np.array(3.0))
y = add(add(x,x), x)
y.backward()
print(x.grad)

# 이어서 두번재 계산 수행
y = add(add(x,x),x)
y.backward()
print(x.grad)

# grad를 초기화 한 뒤의 계산
x.cleargrad()
y = add(add(x,x),x)
y.backward()
print(x.grad)
