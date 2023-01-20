from A_pk import Layer
from A_pk import utils
import A_pk.functions as F
import A_pk.layers as L

#-- plot 기능과 Layer class를 상속받은 Model class 생성
class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

#-- MLP 클래스 정의
class MLP(Model):
    def __init__(self, fc_output_sizes, activation = F.sigmoid):
        '''
        fc_output_sizes (list or tuple) : output size의 개수들이 저장
        activation (Function) : 활성화 함수
        '''
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, '1' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]: 
            x = self.activation(l(x))

        return self.layers[-1](x)  # 마지막 layer는 활성화 함수 수행x


