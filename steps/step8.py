# 여기서는 역전파를 구현하려고 한다.
import numpy as np

class Variable:
    def __init__(self,data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func 

    # 역전파 자동화
    def backward(self):
        print('중간 확인 : ',self.creator)
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input # 이전 입력값 보관
        self.output = output # output 저장

        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

# Square 클래스 정의
class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

print(y.data)

y.grad = np.array(1.0)
y.backward()
print(x.grad)
#b.grad = C.backward(y.grad)
#a.grad = B.backward(b.grad)
#x.grad = A.backward(a.grad)
print(y.creator)
print(b.creator)
print(a.creator)
print(x.creator)
