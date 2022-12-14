# step13에서는 역전파의 가변인자를 활용하려고 한다.
import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, f):
        self.creator = f

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            # 입출력이 여러개일 때를 가정하여 코드 구현
            gys = [output.grad for output in f.outputs]
            print('--- gys 확인 : ', gys)
            print('--- func 확인 : ', f)
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            print('---- gxs 확인 : ', gxs)
            print('--- f.input 확인 ; ', f.inputs)
            for x, gx in zip(f.inputs, gxs):
                x.grad = gx

                if x.creator is not None:
                    funcs.append(x.creator)
                

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs):
        xs = [input.data for input in inputs]
        print('inputs 값 확인 단계 : ', xs)
        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs

        return  outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)

    # 덧셈의 역전파는 그대로 흘러들어온다.
    def backward(self, gy):
        return gy, gy
            
def add(x0, x1):
    return Add()(x0, x1)

# Square 클래스 정의
class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)


x0 = Variable(np.array(2.0))
x1 = Variable(np.array(3.0))

z = add(square(x0), square(x1))
z.backward()

print(z.data)
print(x0.grad)
print(x1.grad)
