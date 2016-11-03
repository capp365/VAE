import pickle
import sys
import numpy as np
from matplotlib import pyplot as plt
import networks.default_encoder as encoder
import networks.Bernoulli_decoder as decoder
sys.path.append('../')
import VAE


mnist=pickle.load(open("./mnist.pkl",'rb'))

mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
batchsize = 100
n_epoch = 100
n_latent = 20


x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

model = VAE.VAE(encoder=encoder.Encoder(784,500,n_latent),decoder=decoder.Decoder(784,500,n_latent),sampling=1, gauss=False)

model.load_files('./ch_cln')

print model.errors[0],model.errors[-1]
plt.plot(model.errors)
plt.show()

# original images and reconstructed images
def save_images(x, filename):
    fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
    for ai, xi in zip(ax.flatten(), x):
        ai.imshow(xi.reshape(28, 28))
    fig.savefig(filename)

train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
x = [j for j in np.asarray(x_train[train_ind])]
x1 = model.reconstruction(x)
save_images(x, 'train')
save_images([j for j in x1], 'train_reconstructed')

test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
x = [j for j in np.asarray(x_test[test_ind])]
x1 = model.reconstruction(x)
save_images(x, 'test')
save_images([j for j in x1], 'test_reconstructed')


## draw images from randomly sampled z
#z = [j for j in np.random.normal(0, 1, (9, n_latent)).astype(np.float32)]
#x = model.sampling(z)
#save_images(x.data, 'sampled')