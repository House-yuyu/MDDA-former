import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
import math
import sys

sys.path.append("..")
from utils.odconv import ODConv2d

def odconv3x3(in_planes, out_planes, stride=1, reduction=0.25, kernel_num=1):
    return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    reduction=reduction, kernel_num=kernel_num)


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)  

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    """
    All bias set false
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm2d(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)

        ffn_channel = ffn_expansion_factor * dim
        self.conv_ff1 = nn.Conv2d(in_channels=dim, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1,
                                  groups=1, bias=False)
        self.dwconv = nn.Conv2d(ffn_channel, ffn_channel, kernel_size=3, stride=1, padding=1, groups=ffn_channel, bias=False)
        self.sg = SimpleGate()
        self.conv_ff2 = nn.Conv2d(in_channels=int(ffn_channel) // 2, out_channels=dim, kernel_size=1, padding=0,
                                  stride=1, groups=1, bias=False)

        self.beta = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1), requires_grad=True)

    def forward(self, x):
        input = x
        y = self.beta * x + self.attn(self.norm1(x))  # rescale

        y = self.conv_ff1(self.norm2(y))
        y = self.dwconv(y)   
        y = self.sg(y)
        y = self.gamma * y + self.conv_ff2(x)

        return y + input


class Localcnn_block(nn.Module):
    def __init__(self, c, DW_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)

        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sg = SimpleGate()
        self.attention = odconv3x3(dw_channel // 2, dw_channel // 2)

        self.norm1 = LayerNorm2d(c)

        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        x = self.attention(x)

        x = self.conv3(x)

        # x = self.dropout1(x)

        # y = inp * self.beta + x
        y = inp + x

        return y


## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Sequential(nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


#########################################################################
class MDDA_former(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=60,
                 num_blocks=[3, 6, 6, 10, 6, 6, 3],   # {dim=48, 2, 6, 8, 10, 4, 3, 2}  {dim=60, 3, 6, 6, 10, 6, 6, 3}
                 heads=[1, 2, 4, 10],  #  {dim=32 or 64, heads[3]=8} {dim=48, heads[3]=12} {dim=60, heads[3]=10}
                 ffn_expansion_factor=2,
                 bias=False):
        super(MDDA_former, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[Localcnn_block(int(dim)) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(
            *[Localcnn_block(int(dim * 2 ** 1)) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(
            *[Localcnn_block(int(dim * 2 ** 2)) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             ) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            Localcnn_block(int(dim * 2 ** 2)) for i in range(num_blocks[4])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[Localcnn_block(int(dim * 2 ** 1)) for i in range(num_blocks[5])])

        self.up2_1 = Upsample(int(dim * 2))  ## From Level 2 to Level 1
        self.decoder_level1 = nn.Sequential(
            *[Localcnn_block(int(dim * 2)) for i in range(num_blocks[6])])

        self.u1_ = nn.Conv2d(dim * 2, dim, 3, 1, 1, bias=bias)
        self.last = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=False)


    def net_update_temperature(self, temperature):  # for Odconv
        for m in self.modules():
            if hasattr(m, "update_temperature"):
                m.update_temperature(temperature)


    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)  #
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)  # 2c
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)  # 4c
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # u4_ = self.u4_(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        # u3_ = self.u3_(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)  # 2c
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)  # 4c
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)  # 2c
        out_dec_level2 = self.decoder_level2(inp_dec_level2)  # FL2

        # u2_ = self.u2_(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)  # c
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)  # 2c
        out_dec_level1 = self.decoder_level1(inp_dec_level1)  # FL1

        u1_ = self.u1_(out_dec_level1)

        output = self.last(u1_) + inp_img

        return output
