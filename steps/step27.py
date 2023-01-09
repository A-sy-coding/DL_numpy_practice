if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from A_pk import Variable
import A_pk.functions as F
import numpy as np

x = Variable(np.array(np.pi/4))
y = F.sin(x)
y.backward()

print(y)
print(y.data)
print(x.grad)

x = Variable(np.array(np.pi/4))
y = F.my_sin(x)
y.backward()

print(y.data)
print(x.grad)


