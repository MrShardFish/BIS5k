import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.modules.layers import *
from utils.utils import *



class CCR(nn.Module):
    def __init__(self, in_channel, channel):
        super(CCR, self).__init__()
        self.channel = channel

        self.conv_query = nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))
        self.conv_key = nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))

        self.conv_out1r = conv(channel, channel, 3, relu=True)
        self.conv_out1g = conv(channel, channel, 3, relu=True)
        self.conv_out1b = conv(channel, channel, 3, relu=True)

        self.conv_out2 = conv(in_channel + channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)

    def forward(self, x):
        x_res = x
        b, c, h, w = x.shape
       # k q v compute
        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        key = self.conv_key(x).view(b, self.channel, -1)
        value = self.conv_value(x).view(b, self.channel, -1).permute(0, 2, 1)

        # attention for channel1
        # compute similarity map
        sim = torch.bmm(query, key)  # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context1 = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context1 = self.conv_out1r(context1)

        # attention for channel2
        # compute similarity map
        sim = torch.bmm(key.permute(0, 2, 1), value.permute(0, 2, 1))  # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context2 = torch.bmm(sim, query).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context2 = self.conv_out1g(context2)

        # attention for channel3
        # compute similarity map
        sim = torch.bmm(value, query.permute(0, 2, 1))  # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context3 = torch.bmm(sim, key.permute(0, 2, 1)).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context3 = self.conv_out1b(context3)

        context = (context1+context2+context3)/3

        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        return x

if __name__ == '__main__':
    model = CCR(256,256)
    model = model.cuda()
    image = torch.randn([1, 256, 88, 88]).cuda()
    output = model(image)
