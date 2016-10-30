import os,sys
import numpy as np
from chainer import Variable, optimizers, serializers
sys.path.append('../')
import VAE

fs = [lambda x: np.random.multivariate_normal(10 * np.ones(2), np.eye(2)),
      lambda x: np.random.multivariate_normal(30 * np.ones(2), np.eye(2)),
      lambda x: np.random.multivariate_normal(50 * np.ones(2), np.eye(2))]

are = np.random.randint(0, 3, 60)
input_data = [fs[i](1) for i in are]

itera_n=10

while(True):
    model = VAE.VAE_core()
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    errors = []

    for i in range(itera_n):
        x=Variable(np.array([np.float32(j) for j in input_data],dtype=np.float32))
        optimizer.update(model,x,100)
        errors.append(model(x,100).data)
        model.cleargrads()
        if np.isnan(errors[-1]):
            print 'NaN break!'
            break
    if len(errors)==itera_n:
        break
print errors[0],errors[-1],min(errors),max(errors)
print model(x,100).data
model.cleargrads()
serializers.save_npz('my.model', model)
model2 = VAE.VAE_core()
print model2(x,100).data
model2.cleargrads()
serializers.load_npz('my.model', model2)
print model2(x,100).data
model2.cleargrads()