import chainer.functions as F
import chainer.links as L
from chainer import Chain


class Decoder(Chain):
    """Default Decoder"""

    def __init__(self, dim_x=2, dim_h=3, dim_z=2):
        super(Decoder, self).__init__(
            dec_l1=L.Linear(dim_z, dim_h),
            dec_l2=L.Linear(dim_h, dim_x),
            dec_l3=L.Linear(dim_h, dim_x),
        )

    def __call__(self, z):
        h1 = F.relu(self.dec_l1(z))
        mu = self.dec_l2(h1)
        sigma = self.dec_l3(h1)
        return mu, sigma