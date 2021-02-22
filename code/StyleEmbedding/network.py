# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3
        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))

        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq + 1e-10)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  
            
            nn.Conv2d(32, 64, 7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  
            
            nn.Conv2d(64, 128, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(128, 256, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
#            nn.BatchNorm2d(256,affine=True),
        )
#        self.liner = nn.Sequential(nn.Linear(86016, 256), nn.Sigmoid())
        self.liner = nn.Sequential(
                nn.Linear(86016, 256), 
                nn.ReLU(inplace=True),
                nn.Linear(256, 64),
#                nn.Linear(64, 2),
#                nn.Sigmoid()
                )
        self.out = nn.Linear(64, 2)
#        self.out = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    def forward_cl(self, x): 
        x = self.conv(x)
        x = x.reshape(x.size()[0], -1)
        x = self.liner(x)
        return x
    
    def forward_one(self, x):
        x = self.conv(x)               # [32, 256, 12, 28]
        x = x.reshape(x.size()[0], -1) # [32, 86016]
        x = self.liner(x)              # [32, 64]
        x = self.out(x)                # [32, 2]
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2