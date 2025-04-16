import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, input):
        out = self.conv1(input)
        out = F.relu(out)
        out = self.conv2(out)
        out = input + out
        return out
class EncoderNet(nn.Module):
    def __init__(self, in_dim=1, conv_dim = 32, out_dim=32, num_blocks=4):
        super(EncoderNet, self).__init__()
        self.inputConv = nn.Conv2d(in_channels=in_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.ResBlocks = nn.Sequential(*[ResBlock(conv_dim) for i in range(num_blocks)])

        self.outputConv = nn.Conv2d(in_channels=conv_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, input):
        out = self.inputConv(input)
        out = self.ResBlocks(out)
        out = F.relu(out)
        out = self.outputConv(out)
        return out
class CoarseFuseNet(nn.Module):
    def __init__(self, in_dim=5, out_dim=32):
        super(CoarseFuseNet, self).__init__()
        self.encoder = EncoderNet(in_dim, out_dim)
        self.Conv = nn.Conv2d(in_channels=64, out_channels=out_dim, kernel_size=3, stride=1, padding=1,padding_mode='reflect')
        self.out_dim = out_dim
    def forward(self, input):
        batch, group_length, group_size, h, w =input.size()
        input = input.view(batch*group_length, group_size, h, w) #
        out = self.encoder(input) # batch*group_length, 32, h, w
        out = out.view(batch, group_length, self.out_dim, h, w)
        cat1_stage1 = torch.cat([out[:,0,...],out[:,1,...]], dim=1)
        cat1_stage1 = self.Conv(cat1_stage1)
        cat2_stage1 = torch.cat([out[:,2,...],out[:,3,...]], dim=1)
        cat2_stage1 = self.Conv(cat2_stage1)
        cat_stage2 = torch.cat([cat1_stage1,cat2_stage1], dim=1)
        return cat_stage2
class Sharpmodule(nn.Module):
    def __init__(self, in_dim=16):
        super(Sharpmodule, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(1, in_dim, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_dim, 1, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, input):
        out = self.Conv(input)
        return out

class ConvLeaky(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvLeaky, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        # nn.init.kaiming_normal_(self.conv1.weight)
        # nn.init.kaiming_normal_(self.conv2.weight)

    def forward(self, input):
        out = self.conv1(input)
        # print('conv1: {}'.format(out.size()))
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        out = F.leaky_relu(out, 0.2)
        return out
class FNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, mode):
        super(FNetBlock, self).__init__()
        self.convleaky = ConvLeaky(in_dim, out_dim)
        if mode == "maxpool":
            self.final = lambda x: F.max_pool2d(x, kernel_size=2)
        elif mode == "bilinear":
            self.final = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        else:
            raise Exception('mode must be maxpool or bilinear')

    def forward(self, input):
        out = self.convleaky(input)
        out = self.final(out)
        return out

class FNet(nn.Module):
    def __init__(self, in_dim=2):
        super(FNet, self).__init__()
        self.convPool1 = FNetBlock(in_dim, 32, mode="maxpool")
        self.convPool2 = FNetBlock(32, 64, mode="maxpool")
        self.convPool3 = FNetBlock(64, 128, mode="maxpool")
        self.convBinl1 = FNetBlock(128, 256, mode="bilinear")
        self.convBinl2 = FNetBlock(256, 128, mode="bilinear")
        self.convBinl3 = FNetBlock(128, 64, mode="bilinear")
        self.seq = nn.Sequential(self.convPool1, self.convPool2, self.convPool3,
                                 self.convBinl1, self.convBinl2, self.convBinl3)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1,
                               padding_mode='reflect')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1,
                               padding_mode='reflect')
        # self.refine = Refiner()

        # nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, input):
        out = self.seq(input)
        out = self.conv1(out)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        # refineout = self.refine(torch.cat((input[:, 0:1, ...], out), dim=1)) # torch.cat((input[:, 1:2, ...], out), dim=1)
        # out = out + refineout
        self.out = torch.tanh(out) * 5.0
        # self.out.retain_grad()
        return self.out




#class DecoderNet(nn.Module):

#class MENet(nn.Module):