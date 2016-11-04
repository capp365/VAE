import numpy as np
import os
import pickle
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import networks.default_encoder as encoder
import networks.default_decoder as decoder
import chainer.links as L
from chainer.training import extensions

class VAE_core(Chain):
    """Pyoko Pyoko Implementation of Variational Auto-Encoder"""

    def __init__(self, encoder=encoder.Encoder(),decoder=decoder.Decoder(),Gauss=False):
        super(VAE_core, self).__init__(encoder=encoder, decoder=decoder)
        self.Gauss =Gauss

    def __call__(self, l=1):
        def lf(x):
            mu,sigma = self.encoder(x)
            #self.KL=
            self.loss = 0
            for i in range(l):
                sample=F.gaussian(mu,sigma)
                if self.Gauss==True:
                    m,s=self.decoder(sample)
                    self.loss += F.gaussian_nll(x,m,s)
                else:
                    p = self.decoder(sample)
                    self.loss += F.bernoulli_nll(x, p)/(l*len(x.data))
            self.KL=F.gaussian_kl_divergence(mu,sigma)/len(x.data)
            self.loss += self.KL
            #self.loss = self.loss
            return self.loss
        return lf

class VAE:
    """User Interface for VAE"""

    def __init__(self,encoder=encoder,decoder=decoder,sampling=100,gauss=True):
        self.model = VAE_core(encoder,decoder, Gauss=gauss)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.errors=[]
        self.kls=[]
        self.sampling = sampling

    def fit(self,input_data,iter=1):
        for i in range(iter):
            x = Variable(np.array([np.float32(j) for j in input_data], dtype=np.float32))
            self.optimizer.update(self.model(), x)
            self.errors.append(self.model()(x).data)
            #self.kls.append(self.model.KL.data)
            #self.model.cleargrads()
            #if np.isnan(self.errors[-1]):
            #    return False
            #else:
            #    return True

    def generate_z(self,input_data):
        x = Variable(np.array([np.float32(j) for j in input_data], dtype=np.float32))
        tmp = self.model.encoder.reconstruct(x).data
        self.model.cleargrads()
        return tmp

    #def sampling(self,num=1):


    def reconstruction(self,input_data):
        x = Variable(np.array([np.float32(j) for j in input_data], dtype=np.float32))
        tmp = self.model.encoder.reconstruct(x) #Sigma
        tmp = self.model.decoder.reconstruct(tmp).data
        self.model.cleargrads()
        return tmp

    def save_files(self, directory, update=False):
        if os.path.exists(directory) and update==False:
            print directory + 'exists (you can overwrite if update=True).'
        else:
            if os.path.exists(directory)==False:
                os.mkdir(directory)

            serializers.save_npz(directory+'/my.model', self.model)
            pickle.dump([self.errors,self.sampling],open(directory+'/params.pickle','wb'))


    def load_files(self, directory):
        if os.path.exists(directory)==False:
            print 'directory does not be found.'
        else:
            serializers.load_npz(directory + '/my.model', self.model)
            self.errors, self.sampling=pickle.load(open(directory + '/params.pickle','rb'))