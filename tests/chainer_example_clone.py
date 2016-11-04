import pickle
import sys
import numpy as np
from chainer import computational_graph
import networks.default_encoder as encoder
import networks.Bernoulli_decoder as decoder
from matplotlib import pyplot as plt
sys.path.append('../')
import VAE
# original images and reconstructed images
def save_images(x, filename):
    fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
    for ai, xi in zip(ax.flatten(), x):
        ai.imshow(xi.reshape(28, 28))
    fig.savefig(filename)


mnist=pickle.load(open("./mnist.pkl",'rb'))

mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

mnist['data'] = mnist['data'][0:100]
mnist['target'] = mnist['target'][0:100]

N = 60000
N = 30

batchsize = 100
n_epoch = 100
n_latent = 20


x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

#model = VAE.VAE(784, n_latent, 500)
model = VAE.VAE(encoder=encoder.Encoder(784,500,n_latent),decoder=decoder.Decoder(784,500,n_latent),sampling=1, gauss=False)

for epoch in range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    for i in range(0, N, batchsize):
        x = [j for j in np.asarray(x_train[perm[i:i + batchsize]])]
        model.fit(x,1)
        train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
        x = np.asarray(x_train[train_ind])
        x1 = model.reconstruction(x)
        save_images(x1, './pics/'+str(epoch))


train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
x = np.asarray(x_train[train_ind])
x1 = model.reconstruction(x)
save_images(x, './pics/train')
save_images(x1, './pics/train_reconstructed')

test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
x = np.asarray(x_test[test_ind])
x1 = model.reconstruction(x)
save_images(x, './pics/test')
save_images(x1, './pics/test_reconstructed')


model.save_files('./ch_cln')