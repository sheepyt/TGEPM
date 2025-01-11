import numpy as np
import torch
import torch.nn as nn

ddac_npy = np.load("./ddac_kernels.npy")
ddac_npy = torch.from_numpy(ddac_npy)
ddac_npy = ddac_npy.type(torch.FloatTensor).cuda()


class DDAC(nn.Module):
    def __init__(self):
        super(DDAC, self).__init__()
        self.ddac = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3),
                              padding=1, stride=(1, 1))
        self.conv = nn.Conv2d(in_channels=8, out_channels=30, kernel_size=(1, 1),
                              padding=0, stride=(1, 1))

        self.TLU = nn.Hardtanh(-10, 10, True)

        self.ddac.weight = torch.nn.Parameter(ddac_npy, requires_grad=True)

    def forward(self, x):
        x = self.ddac(x)
        x = self.conv(x)
        x = self.TLU(x)
        return x