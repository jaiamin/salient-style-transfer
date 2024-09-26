import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weight(layer):
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

    
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        init_weight(self.conv)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RSU(nn.Module):
    def __init__(self, L, C_in, C_out, M):
        super(RSU, self).__init__()
        self.conv = ConvBlock(C_in, C_out)
        
        self.enc = nn.ModuleList([ConvBlock(C_out, M)])
        for i in range(L-2):
            self.enc.append(ConvBlock(M, M))
            
        self.mid = ConvBlock(M, M, dilation=2)
        
        self.dec = nn.ModuleList([ConvBlock(2*M, M) for _ in range(L-2)])
        self.dec.append(ConvBlock(2*M, C_out))
    
        self.downsample = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
    def forward(self, x):
        x = self.conv(x)
        
        out = []
        for i, enc in enumerate(self.enc):
            if i == 0: out.append(enc(x))
            else: out.append(enc(self.downsample(out[i-1])))
            
        y = self.mid(out[-1])
        
        for i, dec in enumerate(self.dec):
            if i > 0: y = self.upsample(y)
            y = dec(torch.cat((out[len(self.dec)-i-1], y), dim=1))
            
        return x + y
    

class RSU4F(nn.Module):
    def __init__(self, C_in, C_out, M):
        super(RSU4F, self).__init__()
        self.conv = ConvBlock(C_in, C_out)
        
        self.enc = nn.ModuleList([
            ConvBlock(C_out, M),
            ConvBlock(M, M, dilation=2),
            ConvBlock(M, M, dilation=4)
        ])
        
        self.mid = ConvBlock(M, M, dilation=8)
        
        self.dec = nn.ModuleList([
            ConvBlock(2*M, M, dilation=4),
            ConvBlock(2*M, M, dilation=2),
            ConvBlock(2*M, C_out)
        ])
        
    def forward(self, x):
        x = self.conv(x)
        
        out = []
        for i, enc in enumerate(self.enc):
            if i == 0: out.append(enc(x))
            else: out.append(enc(out[i-1]))
            
        y = self.mid(out[-1])
        
        for i, dec in enumerate(self.dec):
            y = dec(torch.cat((out[len(self.dec)-i-1], y), dim=1))
            
        return x + y
    
    
class U2Net(nn.Module):
  def __init__(self):
    super(U2Net, self).__init__()
    self.enc = nn.ModuleList([
      RSU(L=7, C_in=3, C_out=64, M=32),
      RSU(L=6, C_in=64, C_out=128, M=32),
      RSU(L=5, C_in=128, C_out=256, M=64),
      RSU(L=4, C_in=256, C_out=512, M=128),
      RSU4F(C_in=512, C_out=512, M=256),
      RSU4F(C_in=512, C_out=512, M=256)
    ])

    self.dec = nn.ModuleList([
      RSU4F(C_in=1024, C_out=512, M=256),
      RSU(L=4, C_in=1024, C_out=256, M=128),
      RSU(L=5, C_in=512, C_out=128, M=64),
      RSU(L=6, C_in=256, C_out=64, M=32),
      RSU(L=7, C_in=128, C_out=64, M=16)
    ])

    self.convs = nn.ModuleList([
      nn.Conv2d(64, 1, 3, padding=1),
      nn.Conv2d(64, 1, 3, padding=1),
      nn.Conv2d(128, 1, 3, padding=1),
      nn.Conv2d(256, 1, 3, padding=1),
      nn.Conv2d(512, 1, 3, padding=1),
      nn.Conv2d(512, 1, 3, padding=1)
    ])

    self.lastconv = nn.Conv2d(6, 1, 1)
    self.downsample = nn.MaxPool2d(2, stride=2)

    init_weight(self.lastconv)
    for conv in self.convs:
      init_weight(conv)

  def upsample(self, x, target):
    return F.interpolate(x, size=target.shape[2:], mode='bilinear')

  def forward(self, x):
    enc_out = []
    for i, enc in enumerate(self.enc):
      if i == 0: enc_out.append(enc(x))
      else: enc_out.append(enc(self.downsample(enc_out[i-1])))

    dec_out = [enc_out[-1]]
    for i, dec in enumerate(self.dec):
      dec_out.append(dec(torch.cat((self.upsample(dec_out[i], enc_out[4-i]), enc_out[4-i]), dim=1)))
    
    side_out = []
    for i, conv in enumerate(self.convs):
      if i == 0: side_out.append(conv(dec_out[5]))
      else: side_out.append(self.upsample(conv(dec_out[5-i]), side_out[0]))
    
    side_out.append(self.lastconv(torch.cat(side_out, dim=1)))       

    return [torch.sigmoid(s.squeeze(1)) for s in side_out]