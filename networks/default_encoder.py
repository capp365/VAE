import chainer.functions as F
import chainer.links as L
from chainer import Chain


class Encoder(Chain):
    """Default Encoder"""

    def __init__(self, dim_x=2, dim_h=3, dim_z=2):
        super(Encoder, self).__init__(
            enc_l1=L.Linear(dim_x, dim_h),
            enc_l2=L.Linear(dim_h, dim_z),
            enc_l3=L.Linear(dim_h, dim_z),
        )

    def __call__(self, x):
        h1 = F.relu(self.enc_l1(x))
        mu = self.enc_l2(h1)
        sigma = self.enc_l3(h1)
        return mu, sigma
