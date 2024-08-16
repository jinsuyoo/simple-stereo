from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math

from core.models.extractor import BasicEncoder


class disparity_regression(nn.Module):
    def __init__(self, max_disp):
        super(disparity_regression, self).__init__()

        self.disp = torch.Tensor(
            np.reshape(np.array(range(max_disp)), [1, max_disp, 1, 1]))
        self.disp = self.disp.cuda()

    def forward(self, x):
        out = torch.sum(x * self.disp.data, 1, keepdim=True)
        return out


class PSMNet(nn.Module):
    def __init__(
        self, 
        max_disp, 
        output_dim=32, 
        norm_fn=None, 
        dropout=0.0, 
        n_downsample=2
    ):
        super(PSMNet, self).__init__()

        self.max_disp = max_disp

        self.encoder = BasicEncoder(
            output_dim=output_dim,
            norm_fn='batch',
            dropout=dropout,
            downsample=n_downsample
        )

        self.dres0 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )


        self.dres1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )


        self.dres2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.dres3 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.dres4 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

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
        """Estimate disparity map from stereo images"""

        # shape of image1: [B, 3, H, W]
        
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous() 

        original_h, original_w = image1.shape[2], image1.shape[3]

        image1_feat = self.encoder(image1)
        image2_feat = self.encoder(image2)

        # shape of image1_feat: [B, 32, H/4, W/4]

        b, c, h, w = image1_feat.shape

        # shape of cost_volume: [B, 2*c, max_disp//4, H/4, W/4]
        cost_volume = torch.zeros(
            [b, 2 * c, self.max_disp // 4, h, w],
            dtype=torch.float32,
            device=image1_feat.device
        )

        for i in range(self.max_disp // 4):
            if i > 0:
                cost_volume[:, :c, i, :,i:] = image1_feat[:,:,:,i:]
                cost_volume[:, c:, i, :,i:] = image2_feat[:,:,:,:-i]
            else:
                cost_volume[:, :c, i, :,:] = image1_feat
                cost_volume[:, c:, i, :,:] = image2_feat
        cost_volume = cost_volume.contiguous()

        # shape of cost0: [B, 32, max_disp//4, H/4, W/4]
        cost0 = self.dres0(cost_volume)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0 
        cost0 = self.dres3(cost0) + cost0 
        cost0 = self.dres4(cost0) + cost0

        # shape of cost: [B, 1, max_disp//4, H/4, W/4]
        cost = self.classifier(cost0)

        # shape of cost: [B, max_disp, H, W]
        cost = F.interpolate(
            cost, [self.max_disp, original_h, original_w], mode='trilinear')
        cost = torch.squeeze(cost, 1)

        # shape of pred: [B, max_disp, H, W]
        pred = F.softmax(cost, dim=1)

        # shape of pred: [B, 1, H, W]
        pred = disparity_regression(self.max_disp)(pred)

        return pred