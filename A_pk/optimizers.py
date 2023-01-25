# 매개변수 갱신을 위한 클래스 정의
import numpy as np
import math

class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        # Model or Layer instance 정의
        self.target = target
        return self

    def update(self):
        # None 이외의 매개변수를 리스트에 담는다.
        params = [p for p in self.target.params() if p.grad is not None]

        # 전처리
        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param) # 매개변수 업데이트
        
    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)

#-- Hook function -> 전처리 함수들 정의
class WeightDecay:
    ''' 기울기 감소 '''
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, params):
        for param in params:
            param.grad.data += self.rate * param.data

class ClipGrad:
    ''' 기울기 clipping'''
    def __init__(self, max_norm):
        self.max_norm = max_norm

    def __call__(self, params):
        total_norm = 0
        for param in params:
            total_norm += (param.grad.data **2).sum()
        total_norm = math.sqrt(float(total_norm))

        rate = self.max_norm / (total_norm + 1e-6)
        if rate < 1:
            for param in params:
                param.grad.data *= rate

#-- SGD class
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data

#-- Momentum class
class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param) # unique한 id값들이 저장된다.

        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v

