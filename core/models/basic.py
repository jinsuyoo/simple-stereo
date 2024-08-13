from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *

class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()

########
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
 
        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
 
        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, image1, image2):
        # shape of image1: [B, 3, H, W]

        original_h, original_w = image1.shape[2], image1.shape[3]

        image1_feat = self.feature_extraction(image1)
        image2_feat = self.feature_extraction(image2)

        # shape of image1_feat: [B, 64, H/4, W/4]

        b, c, h, w = image1_feat.shape
    
        cost_volume = torch.zeros(
            [b, 2*c, self.maxdisp//4, h, w],
            dtype=torch.float32,
            #requires_grad=True,
            device=image1_feat.device
        )
        #import pdb; pdb.set_trace()

        for i in range(self.maxdisp//4):
            if i > 0:
                cost_volume[:, :image1_feat.size()[1], i, :,i:] = image1_feat[:,:,:,i:]
                cost_volume[:, image1_feat.size()[1]:, i, :,i:] = image2_feat[:,:,:,:-i]
            else:
                cost_volume[:, :image1_feat.size()[1], i, :,:] = image1_feat
                cost_volume[:, image1_feat.size()[1]:, i, :,:] = image2_feat
        cost_volume = cost_volume.contiguous()

        cost0 = self.dres0(cost_volume)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0 
        cost0 = self.dres3(cost0) + cost0 
        cost0 = self.dres4(cost0) + cost0

        cost = self.classify(cost0)
        cost = F.upsample(cost, [self.maxdisp, original_h, original_w], mode='trilinear')
        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost)
        pred = disparityregression(self.maxdisp)(pred)

        return pred