#  step35 에서는 tanh 함수를 사용하여 고차미분을 진행한다.

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from A_pk import Variable
import numpy as np
from A_pk.utils import plot_dot_graph
import A_pk.functions as F

x = Variable(np.array(1.0))
y = F.tanh(x)

x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 0

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = 'gx' + str(iters+1)
plot_dot_graph(gx, verbose=False, to_file='tanh.png')
