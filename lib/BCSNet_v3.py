import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.optim.losses import *
from lib.modules.layers import *
from lib.modules.context_module import *
from lib.modules.attention_module import *
from lib.modules.decoder_module import *

from lib.backbones.Res2Net_v1b import res2net50_v1b_26w_4s


class BCSNet_v3(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channels=256, output_stride=16, pretrained=True):
        super(BCSNet_v3, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=pretrained, output_stride=output_stride)

        self.context1 = PAA_f(256, channels)
        self.context2 = PAA_e(512, channels)
        self.context3 = PAA_e(1024, channels)
        self.context4 = PAA_e(2048, channels)

        self.decoder = PAA_d(channels)

        self.decoder4 = PAA_srm4(channels)
        self.decoder3 = PAA_srm(channels)
        self.decoder2 = PAA_srm(channels)
        self.decoder1 = PAA_srm(channels)
        self.decoder0 = PAA_srm0(channels)

        self.attention_CCR1 = CCR(256, 128)
        self.attention_CCR2 = CCR(512, 256)
        self.attention_CCR3 = CCR(1024, 512)
        self.attention_CCR4 = CCR(2048, 1024)

        self.tir1 = TIR(256, 128)
        self.tir2 = TIR(512, 256)
        self.tir3 = TIR(1024, 512)
        self.tir4 = TIR(2048, 1024)

        self.loss_fn = bce_iou_loss

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, sample):
        x = sample['image']
        if 'gt' in sample.keys():
            y = sample['gt']
        else:
            y = None

        base_size = x.shape[-2:]

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x1ccr = self.attention_CCR1(x1)
        x1tir = self.tir1(x1)
        x1 = torch.cat((x1ccr,x1tir),dim=1)+x1

        x2 = self.resnet.layer2(x1)
        x2ccr = self.attention_CCR2(x2)
        x2tir = self.tir2(x2)
        x2 = torch.cat((x2ccr,x2tir),dim=1)+x2

        x3 = self.resnet.layer3(x2)
        x3ccr = self.attention_CCR3(x3)
        x3tir = self.tir3(x3)
        x3 = torch.cat((x3ccr,x3tir),dim=1)+x3

        x4 = self.resnet.layer4(x3)
        x4ccr = self.attention_CCR4(x4)
        x4tir = self.tir4(x4)
        x4 = torch.cat((x4ccr,x4tir),dim=1)+x4

        x1 = self.context1(x1)
        x2 = self.context2(x2)
        x3 = self.context3(x3)
        x4 = self.context4(x4)

        # f3, a3 = self.attention3(torch.cat([x3, self.ret(f4, x3)], dim=1), a4)
        # out3 = self.res(a3, base_size)
        #
        # _, a2 = self.attention2(torch.cat([x2, self.ret(f3, x2)], dim=1), a3)
        # out2 = self.res(a2, base_size)

        f4srm, a4srm = self.decoder4(x4, x3, x2, x1)
        out4drm = self.res(a4srm, base_size)

        f3srm, a3srm = self.decoder3(x4, x3, x2, x1, f4srm)
        out3drm = self.res(a3srm, base_size)

        f2srm, a2srm = self.decoder2(x4, x3, x2, x1, f3srm)
        out2drm = self.res(a2srm, base_size)

        f1srm, a1srm = self.decoder1(x4, x3, x2, x1, f2srm)
        out1drm = self.res(a1srm, base_size)

        _, a0srm = self.decoder0(f1srm,f2srm,f3srm,f4srm)
        out0drm = self.res(a0srm, base_size)




        if y is not None:
            loss5 = self.loss_fn(out4drm, y)
            loss4 = self.loss_fn(out3drm, y)
            loss3 = self.loss_fn(out2drm, y)
            loss2 = self.loss_fn(out1drm, y)
            loss1 = self.loss_fn(out0drm, y)

            loss = loss1 + loss2 + loss3 + loss4 + loss5
            debug = [out0drm,out1drm,out2drm,out3drm,out4drm]
        else:
            loss = 0
            debug = []

        return {'pred': out0drm, 'loss': loss, 'debug': debug}

if __name__ == '__main__':
    model = BCSNet_v3()
    model = model.cuda()
    image = torch.randn([1, 3, 128, 128]).cuda()
    gt = torch.randn([1, 3, 128, 128]).cuda()
    input =  {'image':image,'gt':gt}
    output = model(input)