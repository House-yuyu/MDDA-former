from ptflops import get_model_complexity_info
from thop import profile


##########################################################################
## U-Net
bn = 2  # block number-1

import torch
import torch.nn as nn
##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)
    return layer


def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

##########################################################################
## Supervised Attention Module (RAM)
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img

##########################################################################
## Spatial Attention
class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        return x * y

# Spatial Attention Block (SAB)
class SAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(SAB, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)
        self.SA = SALayer(kernel_size=7)

    def forward(self, x):
        res = self.body(x)
        res = self.SA(res)
        res += x
        return res

##########################################################################
## Pixel Attention
class PALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias), # channel <-> 1
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

## Pixel Attention Block (PAB)
class PAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(PAB, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.PA = PALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.PA(res)
        res += x
        return res

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, block):
        super(Encoder, self).__init__()
        if block == 'CAB':
            self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.encoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
        elif block == 'PAB':
            self.encoder_level1 = [PAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.encoder_level2 = [PAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.encoder_level3 = [PAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
        elif block == 'SAB':
            self.encoder_level1 = [SAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.encoder_level2 = [SAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.encoder_level3 = [SAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, x):
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, block):
        super(Decoder, self).__init__()
        if block == 'CAB':
            self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.decoder_level3 = [CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
        elif block == 'PAB':
            self.decoder_level1 = [PAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.decoder_level2 = [PAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.decoder_level3 = [PAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
        elif block == 'SAB':
            self.decoder_level1 = [SAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.decoder_level2 = [SAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
            self.decoder_level3 = [SAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act) for _ in range(bn)]
        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)
        if block == 'CAB':
            self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
            self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        if block == 'PAB':
            self.skip_attn1 = PAB(n_feat, kernel_size, reduction, bias=bias, act=act)
            self.skip_attn2 = PAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        if block == 'SAB':
            self.skip_attn1 = SAB(n_feat, kernel_size, reduction, bias=bias, act=act)
            self.skip_attn2 = SAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        return [dec1, dec2, dec3]

##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

##########################################################################
# Mixed Residual Module
class Mix(nn.Module):
    def __init__(self, m=1):
        super(Mix, self).__init__()
        w = nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2, feat3):
        factor = self.mix_block(self.w)
        other = (1 - factor)/2
        output = fea1 * other.expand_as(fea1) + fea2 * factor.expand_as(fea2) + feat3 * other.expand_as(feat3)
        return output, factor

##########################################################################
# Architecture
class CMFNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=96, scale_unetfeats=48, kernel_size=3, reduction=4, bias=False):
        super(CMFNet, self).__init__()

        p_act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat // 2, kernel_size, bias=bias), p_act,
                                           conv(n_feat // 2, n_feat, kernel_size, bias=bias))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat // 2, kernel_size, bias=bias), p_act,
                                           conv(n_feat // 2, n_feat, kernel_size, bias=bias))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat // 2, kernel_size, bias=bias), p_act,
                                           conv(n_feat // 2, n_feat, kernel_size, bias=bias))

        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'CAB')
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'CAB')

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'PAB')
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'PAB')

        self.stage3_encoder = Encoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'SAB')
        self.stage3_decoder = Decoder(n_feat, kernel_size, reduction, p_act, bias, scale_unetfeats, 'SAB')

        self.sam1o = SAM(n_feat, kernel_size=3, bias=bias)
        self.sam2o = SAM(n_feat, kernel_size=3, bias=bias)
        self.sam3o = SAM(n_feat, kernel_size=3, bias=bias)

        self.mix = Mix(1)
        self.add123 = conv(out_c, out_c, kernel_size, bias=bias)
        self.concat123 = conv(n_feat*3, n_feat, kernel_size, bias=bias)
        self.tail = conv(n_feat, out_c, kernel_size, bias=bias)


    def forward(self, x):
        ## Compute Shallow Features
        shallow1 = self.shallow_feat1(x)
        shallow2 = self.shallow_feat2(x)
        shallow3 = self.shallow_feat3(x)

        ## Enter the UNet-CAB
        x1 = self.stage1_encoder(shallow1)
        x1_D = self.stage1_decoder(x1)
        ## Apply SAM
        x1_out, x1_img = self.sam1o(x1_D[0], x)

        ## Enter the UNet-PAB
        x2 = self.stage2_encoder(shallow2)
        x2_D = self.stage2_decoder(x2)
        ## Apply SAM
        x2_out, x2_img = self.sam2o(x2_D[0], x)

        ## Enter the UNet-SAB
        x3 = self.stage3_encoder(shallow3)
        x3_D = self.stage3_decoder(x3)
        ## Apply SAM
        x3_out, x3_img = self.sam3o(x3_D[0], x)

        ## Aggregate SAM features of Stage 1, Stage 2 and Stage 3
        mix_r = self.mix(x1_img, x2_img, x3_img)
        mixed_img = self.add123(mix_r[0])

        ## Concat SAM features of Stage 1, Stage 2 and Stage 3
        concat_feat = self.concat123(torch.cat([x1_out, x2_out, x3_out], 1))
        x_final = self.tail(concat_feat)

        return [x_final + mixed_img, mixed_img, mix_r[1], x1_img, x2_img, x3_img, x_final]


if __name__ == "__main__":
    import time
    model = CMFNet()

    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)  # 输入是元组，且不需要batch
    print('flops: ', flops, 'params: ', params)
