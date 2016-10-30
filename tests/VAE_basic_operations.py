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


print 'VAE Initialized.'
while(True):
    pyoriko = VAE.VAE(sampling=50)
    pyoriko.fit(input_data,10)
    error_is_not_nan=pyoriko.fit(input_data,20)
    if error_is_not_nan==True:
        break

print 'Fit method works.'
pyoriko.generate_z(input_data)
print 'generation works.'
pyoriko.reconstruction(input_data)
print 'reconstruction works.'
pyoriko.save_files('./yuno')

pyoriko2 = VAE.VAE()
pyoriko2.load_files('./yuno')
print pyoriko2.sampling
pyoriko2.generate_z(input_data)
pyoriko2.reconstruction(input_data)
print 'save & load works.'