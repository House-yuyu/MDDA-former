import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from torch.nn import init
import functools
import math
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair
# from deconv import FastDeconv
# from DCNv2.dcn_v2 import DCN


def isqrt_newton_schulz_autograd(A, numIters):
    dim = A.shape[0]
    normA=A.norm()
    Y = A.div(normA)
    I = torch.eye(dim,dtype=A.dtype,device=A.device)
    Z = torch.eye(dim,dtype=A.dtype,device=A.device)

    for i in range(numIters):
        T = 0.5*(3.0*I - Z@Y)
        Y = Y@T
        Z = T@Z
    #A_sqrt = Y*torch.sqrt(normA)
    A_isqrt = Z / torch.sqrt(normA) # 取的是平方根的倒数
    return A_isqrt

def isqrt_newton_schulz_autograd_batch(A, numIters):
    batchSize,dim,_ = A.shape
    normA=A.view(batchSize, -1).norm(2, 1).view(batchSize, 1, 1)
    Y = A.div(normA)
    I = torch.eye(dim,dtype=A.dtype,device=A.device).unsqueeze(0).expand_as(A)
    Z = torch.eye(dim,dtype=A.dtype,device=A.device).unsqueeze(0).expand_as(A)

    for i in range(numIters):
        T = 0.5*(3.0*I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    #A_sqrt = Y*torch.sqrt(normA)
    A_isqrt = Z / torch.sqrt(normA)

    return A_isqrt


#deconvolve channels
class ChannelDeconv(nn.Module):
    def __init__(self,  block, eps=1e-2,n_iter=5,momentum=0.1,sampling_stride=3):
        super(ChannelDeconv, self).__init__()

        self.eps = eps
        self.n_iter=n_iter
        self.momentum=momentum
        self.block = block

        self.register_buffer('running_mean1', torch.zeros(block, 1))
        #self.register_buffer('running_cov', torch.eye(block))
        self.register_buffer('running_deconv', torch.eye(block))
        self.register_buffer('running_mean2', torch.zeros(1, 1))
        self.register_buffer('running_var', torch.ones(1, 1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.sampling_stride=sampling_stride
    def forward(self, x):
        x_shape = x.shape
        if len(x.shape)==2:
            x=x.view(x.shape[0],x.shape[1],1,1)
        if len(x.shape)==3:
            print('Error! Unsupprted tensor shape.')

        N, C, H, W = x.size()
        B = self.block

        #take the first c channels out for deconv
        c=int(C/B)*B
        if c==0:
            print('Error! block should be set smaller.')

        #step 1. remove mean
        if c!=C:
            x1=x[:,:c].permute(1,0,2,3).contiguous().view(B,-1)
        else:
            x1=x.permute(1,0,2,3).contiguous().view(B,-1)

        if self.sampling_stride > 1 and H >= self.sampling_stride and W >= self.sampling_stride:
            x1_s = x1[:,::self.sampling_stride**2]
        else:
            x1_s=x1

        mean1 = x1_s.mean(-1, keepdim=True)

        if self.num_batches_tracked==0:
            self.running_mean1.copy_(mean1.detach())
        if self.training:
            self.running_mean1.mul_(1-self.momentum)
            self.running_mean1.add_(mean1.detach()*self.momentum)
        else:
            mean1 = self.running_mean1

        x1=x1-mean1

        #step 2. calculate deconv@x1 = cov^(-0.5)@x1
        if self.training:
            cov = x1_s @ x1_s.t() / x1_s.shape[1] + self.eps * torch.eye(B, dtype=x.dtype, device=x.device)
            deconv = isqrt_newton_schulz_autograd(cov, self.n_iter)

        if self.num_batches_tracked==0:
            #self.running_cov.copy_(cov.detach())
            self.running_deconv.copy_(deconv.detach())

        if self.training:
            #self.running_cov.mul_(1-self.momentum)
            #self.running_cov.add_(cov.detach()*self.momentum)
            self.running_deconv.mul_(1 - self.momentum)
            self.running_deconv.add_(deconv.detach() * self.momentum)
        else:
            # cov = self.running_cov
            deconv = self.running_deconv

        x1 =deconv@x1

        #reshape to N,c,J,W
        x1 = x1.view(c, N, H, W).contiguous().permute(1,0,2,3)

        # normalize the remaining channels
        if c!=C:
            x_tmp=x[:, c:].view(N,-1)
            if self.sampling_stride > 1 and H>=self.sampling_stride and W>=self.sampling_stride:
                x_s = x_tmp[:, ::self.sampling_stride ** 2]
            else:
                x_s = x_tmp

            mean2=x_s.mean()
            var=x_s.var()

            if self.num_batches_tracked == 0:
                self.running_mean2.copy_(mean2.detach())
                self.running_var.copy_(var.detach())

            if self.training:
                self.running_mean2.mul_(1 - self.momentum)
                self.running_mean2.add_(mean2.detach() * self.momentum)
                self.running_var.mul_(1 - self.momentum)
                self.running_var.add_(var.detach() * self.momentum)
            else:
                mean2 = self.running_mean2
                var = self.running_var

            x_tmp = (x[:, c:] - mean2) / (var + self.eps).sqrt()
            x1 = torch.cat([x1, x_tmp], dim=1)


        if self.training:
            self.num_batches_tracked.add_(1)

        if len(x_shape)==2:
            x1=x1.view(x_shape)
        return x1

#An alternative implementation
class Delinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=512):
        super(Delinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()



        if block > in_features:
            block = in_features
        else:
            if in_features%block!=0:
                block=math.gcd(block,in_features)
                print('block size set to:', block)
        self.block = block
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(self.block))
        self.register_buffer('running_deconv', torch.eye(self.block))


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        if self.training:

            # 1. reshape
            X=input.view(-1, self.block)

            # 2. subtract mean
            X_mean = X.mean(0)
            X = X - X_mean.unsqueeze(0)
            self.running_mean.mul_(1 - self.momentum)
            self.running_mean.add_(X_mean.detach() * self.momentum)

            # 3. calculate COV, COV^(-0.5), then deconv
            # Cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
            Id = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
            Cov = torch.addmm(self.eps, Id, 1. / X.shape[0], X.t(), X)
            deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)
            # track stats for evaluation
            self.running_deconv.mul_(1 - self.momentum)
            self.running_deconv.add_(deconv.detach() * self.momentum)

        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        w = self.weight.view(-1, self.block) @ deconv
        if self.bias is None:
            b = - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
        else:
            b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)

        w = w.view(self.weight.shape)
        return F.linear(input, w, b)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class FastDeconv(conv._ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,groups=1,bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=64, sampling_stride=3,freeze=False,freeze_iter=100):
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.counter=0
        self.track_running_stats=True
        super(FastDeconv, self).__init__(
            in_channels, out_channels,  _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation),
            False, _pair(0), groups, bias, padding_mode='zeros')

        if block > in_channels:
            block = in_channels
        else:
            if in_channels%block!=0:
                block=math.gcd(block,in_channels)

        if groups>1:
            #grouped conv
            block=in_channels//groups

        self.block=block

        self.num_features = kernel_size**2 *block
        # for debug
        #print(f'[debug] self.num_features={self.num_features}')
        #exit()

        if groups==1:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_deconv', torch.eye(self.num_features))
        else:
            self.register_buffer('running_mean', torch.zeros(kernel_size ** 2 * in_channels))
            self.register_buffer('running_deconv', torch.eye(self.num_features).repeat(in_channels // block, 1, 1))

        self.sampling_stride=sampling_stride*stride
        self.counter=0
        self.freeze_iter=freeze_iter
        self.freeze=freeze

    def forward(self, x):
        N, C, H, W = x.shape
        B = self.block
        frozen=self.freeze and (self.counter>self.freeze_iter)
        if self.training and self.track_running_stats:
            self.counter+=1
            self.counter %= (self.freeze_iter * 10)

        if self.training and (not frozen):

            # 1. im2col: N x cols x pixels -> N*pixles x cols
            if self.kernel_size[0]>1:
                X = torch.nn.functional.unfold(x, self.kernel_size,self.dilation,self.padding,self.sampling_stride).transpose(1, 2).contiguous()
            else:
                #channel wise
                X = x.permute(0, 2, 3, 1).contiguous().view(-1, C)[::self.sampling_stride**2,:]

            if self.groups==1:
                # (C//B*N*pixels,k*k*B)
                X = X.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1, self.num_features)
            else:
                X=X.view(-1,X.shape[-1])

            # 2. subtract mean
            X_mean = X.mean(0)
            X = X - X_mean.unsqueeze(0)

            # 3. calculate COV, COV^(-0.5), then deconv
            if self.groups==1:
                #Cov = X.t() @ X / X.shape[0] + self.eps * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Id=torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Cov = torch.addmm(self.eps, Id, 1. / X.shape[0], X.t(), X)
                deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)
            else:
                X = X.view(-1, self.groups, self.num_features).transpose(0, 1)
                Id = torch.eye(self.num_features, dtype=X.dtype, device=X.device).expand(self.groups, self.num_features, self.num_features)
                Cov = torch.baddbmm(self.eps, Id, 1. / X.shape[1], X.transpose(1, 2), X)

                deconv = isqrt_newton_schulz_autograd_batch(Cov, self.n_iter)

            if self.track_running_stats:
                self.running_mean.mul_(1 - self.momentum)
                self.running_mean.add_(X_mean.detach() * self.momentum)
                # track stats for evaluation
                self.running_deconv.mul_(1 - self.momentum)
                self.running_deconv.add_(deconv.detach() * self.momentum)

        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        #4. X * deconv * conv = X * (deconv * conv)
        if self.groups==1:
            w = self.weight.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1,self.num_features) @ deconv
            b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
            w = w.view(-1, C // B, self.num_features).transpose(1, 2).contiguous()
        else:
            w = self.weight.view(C//B, -1,self.num_features)@deconv
            b = self.bias - (w @ (X_mean.view( -1,self.num_features,1))).view(self.bias.shape)

        w = w.view(self.weight.shape)
        x= F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

        return x



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(DehazeBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


# class DCNBlock(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(DCNBlock, self).__init__()
#         self.dcn = DCN(in_channel, out_channel, kernel_size=(3,3), stride=1, padding=1).cuda()
#     def forward(self, x):
#         return self.dcn(x)


class DCNBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DCNBlock, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # conv则是实际进行的卷积操作，注意这里步长设置为卷积核大小，因为与该卷积核进行卷积操作的特征图是由输出特征图中每个点扩展为其对应卷积核那么多个点后生成的。
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # p_conv是生成offsets所使用的卷积，输出通道数为卷积核尺寸的平方的2倍，代表对应卷积核每个位置横纵坐标都有偏移量。
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation  # modulation是可选参数,若设置为True,那么在进行卷积操作时,对应卷积核的每个位置都会分配一个权重。
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        # 由于卷积核中心点位置是其尺寸的一半，于是中心点向左（上）方向移动尺寸的一半就得到起始点，向右（下）方向移动另一半就得到终止点
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # p0_y、p0_x就是输出特征图每点映射到输入特征图上的纵、横坐标值。
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    # 输出特征图上每点（对应卷积核中心）加上其对应卷积核每个位置的相对（横、纵）坐标后再加上自学习的（横、纵坐标）偏移量。
    # p0就是将输出特征图每点对应到卷积核中心，然后映射到输入特征图中的位置；
    # pn则是p0对应卷积核每个位置的相对坐标；
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        # 计算双线性插值点的4邻域点对应的权重
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class Dehaze(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, padding_type='reflect'):
        super(Dehaze, self).__init__()

        ###### downsample
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                   nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))

        ###### FFA blocks
        self.block = DehazeBlock(default_conv, ngf * 4, 3)

        ###### upsample
        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())


        self.dcn_block = DCNBlock(256, 256)

        self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)

        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)

    def forward(self, input):

        x_deconv = self.deconv(input) # preprocess

        x_down1 = self.down1(x_deconv) # [bs, 64, 256, 256]
        x_down2 = self.down2(x_down1) # [bs, 128, 128, 128]
        x_down3 = self.down3(x_down2) # [bs, 256, 64, 64]

        x1 = self.block(x_down3)
        x2 = self.block(x1)
        x3 = self.block(x2)
        x4 = self.block(x3)
        x5 = self.block(x4)
        x6 = self.block(x5)

        x_dcn1 = self.dcn_block(x6)
        x_dcn2 = self.dcn_block(x_dcn1)

        x_out_mix = self.mix1(x_down3, x_dcn2)
        x_up1 = self.up1(x_out_mix) # [bs, 128, 128, 128]
        x_up1_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_up1_mix) # [bs, 64, 256, 256]
        out = self.up3(x_up2) # [bs,  3, 256, 256]

        return out


if __name__ == "__main__":

    model = Dehaze(3,3)

    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)  # 输入是元组，且不需要batch
    print('flops: ', flops, 'params: ', params)

    # 47.9G  2.61M