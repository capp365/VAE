import pickle
import sys
import numpy as np
import networks.default_encoder as encoder
import networks.Bernoulli_decoder as decoder
from matplotlib import pyplot as plt
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

#model = VAE.VAE(784, n_latent, 500)
model = VAE.VAE(encoder=encoder.Encoder(784,500,n_latent),decoder=decoder.Decoder(784,500,n_latent),sampling=1, gauss=False)

for epoch in range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    for i in range(0, N, batchsize):
        x = [j for j in np.asarray(x_train[perm[i:i + batchsize]])]
        model.fit(x,1)
    fig =plt.figure()
    plt.matshow(model.model.encoder.enc_l1.W.data)
    plt.savefig('./'+str(epoch)+'.png')

model.save_files('./ch_cln')