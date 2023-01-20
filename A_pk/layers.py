from A_pk.core_simple import Parameter
import weakref
import numpy as np
import A_pk.functions as F

#-- 변수 변환 클래스 -> 매개변수 저장
class Layer:
    def __init__(self):
        self._params = set()  # parameter 저장 공간

    def __setattr__(self, name, value):
        ''' value가 Parameter instance이면 저장한다.'''
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        ''' 호출시 forward를 진행한다.'''
        outputs = self.forward(*inputs)

        if not isinstance(outputs, tuple):
            outputs = (outputs, )

        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        ''' Parameter instance 꺼내기'''

        for name in self._params:
            obj =  self.__dict__[name]
            
            # Layer 안에 Layer를 만들고 해당 Layer 안에서 Parameter 클래스 정의
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj


    def cleargrads(self):
        ''' 모든 매개변수 기울기 재설정'''
        for param in self.params():
            #print('param type 확인 : ', type(param))
            param.cleargrad()

#-- Layer를 상속받는 Linear class
#class Linear(Layer):
#    def __init__(self, in_size, out_size, nobias=False, dtype=np.float32):
#        super().__init__()
#
#        I, O = in_size, out_size
#        W_data = np.random.randn(I, O).astype(dtype) * np.sqrt(1 / I) # I*O 형태를 가진다.
#        self.W = Parameter(W_data, name='W') # Variable을 상속받으므로 인자값이 data, name이 들어가게 된다.
#        
#        if nobias:
#            self.b = None
#        else:
#            self.b = Parameter(np.zeros(O, dtype=dtype), name='b')
#
#    def forward(self, x):
#        y = F.linear(x, self.W, self.b)
#        return y

#-- 개선된 Linear class -> in_size를 자동으로 결정하게 만든다. -> 사용성이 좋아짐
class Linear(Layer):
    ''' 가중치 W를 forward를 수행할 때 초기화 되도록 구현한다.'''
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size = None):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')

        if self.in_size is not None:
            self._init_W()  # _init_W 함수 호출

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y


