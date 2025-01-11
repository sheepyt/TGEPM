import torch
from timm.models.layers import trunc_normal_
from SRM_DDAC.getSRM_DDAC import *
from others import *


class SELayer(nn.Module):
    def __init__(self, channel, reduction=5):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Processing(nn.Module):
    def __init__(self):
        super(Processing, self).__init__()

        self.ddac = DDAC()
        channel = 30

        self.Resnet_50_sub1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1), padding=0, stride=(1, 1)),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), padding=1, stride=(1, 1)),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

        self.selayer = SELayer(channel)
        self.Resnet_50_sub2 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1), padding=0, stride=(1, 1)),
            nn.BatchNorm2d(channel),
        )
        self.Resnet_50_relu = nn.ReLU(inplace=True)

        self.Resnet_18_1 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), padding=1, stride=(1, 1)),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), padding=1, stride=(1, 1)),
            nn.BatchNorm2d(channel),
        )
        self.Resnet_18_1_relu = nn.ReLU(inplace=True)

        self.Resnet_18_2 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), padding=1, stride=(1, 1)),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), padding=1, stride=(1, 1)),
            nn.BatchNorm2d(channel),
        )
        self.Resnet_18_2_relu = nn.ReLU(inplace=True)

        self.apgm1 = APGM(30, 1)
        self.apgm2 = APGM(30, 1)
        self.apgm3 = APGM(30, 1)

    def forward(self, x, x_map):
        ddacx = self.ddac(x) 
        ddacx = self.apgm1(ddacx, x_map)
        p1 = ddacx 

        x = ddacx + self.Resnet_50_sub2(self.selayer(self.Resnet_50_sub1(ddacx)))
        x = self.Resnet_50_relu(x)
        x = self.apgm2(x, x_map)
        p2 = x 

        x = x + self.Resnet_18_1(x)
        x = self.Resnet_18_1_relu(x)
        x = self.apgm3(x, x_map)
        p3 = x  

        x = x + self.Resnet_18_2(x)
        x = self.Resnet_18_2_relu(x)

        x = torch.cat([p1, p2, p3, x], dim=1) 

        return x


class Model(nn.Module):
    def __init__(self, num_classes=2, embed_dim=64, ape=False):
        super().__init__()

        self.processing = Processing()
        self.num_classes = num_classes

        self.embed_dim = embed_dim
        self.ape = ape

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(1024, num_classes)
        self.fc = nn.Linear(120, 1024)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)
        self.pmgm = PMGM()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        prob_map = self.pmgm(x)
        x = self.processing(x, prob_map) 
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.head(x)
        return x