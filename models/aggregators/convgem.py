import torch
import torch.nn.functional as F
import torch.nn as nn

class GeM(nn.Module):
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class ConvGeM(nn.Module):
    """Implementation of ConvGeM as of <mettere link a qualcosa>

    Args:
        in_channels (int): number of channels in the input of ConvGeM
        out_channels (int, optional): number of channels that ConvGeM outputs. Defaults to 512.
    """
    
    def __init__(self, in_channels, out_channels=512, p=3, eps=1e-6):
        super(ConvGeM, self).__init__()
        self.channel_pool = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)
        self.gem = GeM(p, eps);
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.channel_pool(x)
        x = self.gem(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    

if __name__ == '__main__':
    x = torch.randn(64, 512, 7, 7)
    m = ConvGeM(512, 512, 3)
    r = m(x)
    print(r.shape)