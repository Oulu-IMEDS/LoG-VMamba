import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqConv3x3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super(SeqConv3x3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1x1-conv3x3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = nn.Conv3d(self.inp_planes, self.mid_planes, kernel_size=1, stride=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = nn.Conv3d(self.mid_planes, self.out_planes, kernel_size=3, stride=1, padding=1)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
            
        elif self.type == 'conv1x1x1-sobelx':
            conv0 = nn.Conv3d(self.inp_planes, self.out_planes, kernel_size=1, stride=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0, 0] = -1.0
                self.mask[i, 0, 0, 1, 0] = -2.0
                self.mask[i, 0, 0, 2, 0] = -1.0
                self.mask[i, 0, 0, 0, 2] = 1.0
                self.mask[i, 0, 0, 1, 2] = 2.0
                self.mask[i, 0, 0, 2, 2] = 1.0

                self.mask[i, 0, 1, 0, 0] = -2.0
                self.mask[i, 0, 1, 1, 0] = -4.0
                self.mask[i, 0, 1, 2, 0] = -2.0
                self.mask[i, 0, 1, 0, 2] = 2.0
                self.mask[i, 0, 1, 1, 2] = 4.0
                self.mask[i, 0, 1, 2, 2] = 2.0

                self.mask[i, 0, 2, 0, 0] = -1.0
                self.mask[i, 0, 2, 1, 0] = -2.0
                self.mask[i, 0, 2, 2, 0] = -1.0
                self.mask[i, 0, 2, 0, 2] = 1.0
                self.mask[i, 0, 2, 1, 2] = 2.0
                self.mask[i, 0, 2, 2, 2] = 1.0
                
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1x1-sobely':
            conv0 = nn.Conv3d(self.inp_planes, self.out_planes, kernel_size=1, stride=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0, 0] = -1.0
                self.mask[i, 0, 0, 0, 1] = -2.0
                self.mask[i, 0, 0, 0, 2] = -1.0
                self.mask[i, 0, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2, 1] = 2.0
                self.mask[i, 0, 0, 2, 2] = 1.0

                self.mask[i, 0, 1, 0, 0] = -2.0
                self.mask[i, 0, 1, 0, 1] = -4.0
                self.mask[i, 0, 1, 0, 2] = -2.0
                self.mask[i, 0, 1, 2, 0] = 2.0
                self.mask[i, 0, 1, 2, 1] = 4.0
                self.mask[i, 0, 1, 2, 2] = 2.0

                self.mask[i, 0, 2, 0, 0] = -1.0
                self.mask[i, 0, 2, 0, 1] = -2.0
                self.mask[i, 0, 2, 0, 2] = -1.0
                self.mask[i, 0, 2, 2, 0] = 1.0
                self.mask[i, 0, 2, 2, 1] = 2.0
                self.mask[i, 0, 2, 2, 2] = 1.0
            
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        
        elif self.type == 'conv1x1x1-sobelz':
            conv0 = nn.Conv3d(self.inp_planes, self.out_planes, kernel_size=1, stride=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0, 0] = -1.0
                self.mask[i, 0, 0, 0, 1] = -2.0
                self.mask[i, 0, 0, 0, 2] = -1.0
                self.mask[i, 0, 0, 1, 0] = -2.0
                self.mask[i, 0, 0, 1, 1] = -4.0
                self.mask[i, 0, 0, 1, 2] = -2.0
                self.mask[i, 0, 0, 2, 0] = -1.0
                self.mask[i, 0, 0, 2, 1] = -2.0
                self.mask[i, 0, 0, 2, 2] = -1.0


                self.mask[i, 0, 2, 0, 0] = 1.0
                self.mask[i, 0, 2, 0, 1] = 2.0
                self.mask[i, 0, 2, 0, 2] = 1.0
                self.mask[i, 0, 2, 1, 0] = 2.0
                self.mask[i, 0, 2, 1, 1] = 4.0
                self.mask[i, 0, 2, 1, 2] = 2.0
                self.mask[i, 0, 2, 2, 0] = 1.0
                self.mask[i, 0, 2, 2, 1] = 2.0
                self.mask[i, 0, 2, 2, 2] = 1.0
            
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
            

        elif self.type == 'conv1x1x1-laplacian':
            conv0 = nn.Conv3d(self.inp_planes, self.out_planes, kernel_size=1, stride=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1, 1] = 1.0
                
                self.mask[i, 0, 1, 1, 0] = 1.0
                self.mask[i, 0, 1, 0, 1] = 1.0
                self.mask[i, 0, 1, 1, 1] = -6.0
                self.mask[i, 0, 1, 2, 1] = 1.0
                self.mask[i, 0, 1, 1, 2] = 1.0
                
                self.mask[i, 0, 2, 1, 1] = 1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        if self.type == 'conv1x1x1-conv3x3x3':
            # conv-1x1x1
            y0 = F.conv3d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1, 1) # B C H W D
            y0[:, :, 0:1, :, :] = b0_pad
            y0[:, :, -1:, :, :] = b0_pad
            y0[:, :, :, 0:1, :] = b0_pad
            y0[:, :, :, -1:, :] = b0_pad
            y0[:, :, :, :, 0:1] = b0_pad
            y0[:, :, :, :, -1:] = b0_pad
            # conv-3x3x3
            y1 = F.conv3d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv3d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1, 1) # B C H W D
            y0[:, :, 0:1, :, :] = b0_pad
            y0[:, :, -1:, :, :] = b0_pad
            y0[:, :, :, 0:1, :] = b0_pad
            y0[:, :, :, -1:, :] = b0_pad
            y0[:, :, :, :, 0:1] = b0_pad
            y0[:, :, :, :, -1:] = b0_pad
            # conv-3x3x3
            y1 = F.conv3d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1
    
    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1x1-conv3x3x3':
            # re-param conv kernel
            RK = F.conv3d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3, 4))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, 3, device=device) * self.b0.view(1, -1, 1, 1, 1)
            RB = F.conv3d(input=RB, weight=self.k1).view(-1,) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :, :] = tmp[i, 0, :, :, :]
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv3d(input=k1, weight=self.k0.permute(1, 0, 2, 3, 4))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, 3, device=device) * self.b0.view(1, -1, 1, 1, 1)
            RB = F.conv3d(input=RB, weight=k1).view(-1,) + b1
        return RK, RB


class ECB(nn.Module):
    def __init__(self, dim, depth_multiplier=2, act_type='linear', with_idt=False, permute=True, **kwargs):
        super(ECB, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = dim
        self.out_planes = dim
        self.act_type = act_type
        self.permute = permute
        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv3x3x3 = nn.Conv3d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv1x1x1_3x3x3 = SeqConv3x3x3('conv1x1x1-conv3x3x3', self.inp_planes, self.out_planes, self.depth_multiplier)
        self.conv1x1x1_sbx = SeqConv3x3x3('conv1x1x1-sobelx', self.inp_planes, self.out_planes, -1)
        self.conv1x1x1_sby = SeqConv3x3x3('conv1x1x1-sobely', self.inp_planes, self.out_planes, -1)
        self.conv1x1x1_sbz = SeqConv3x3x3('conv1x1x1-sobelz', self.inp_planes, self.out_planes, -1)
        self.conv1x1x1_lpl = SeqConv3x3x3('conv1x1x1-laplacian', self.inp_planes, self.out_planes, -1)
        
        self.norm = nn.LayerNorm(normalized_shape=dim, elementwise_affine=False)
        
        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.permute:
            x = x.permute(0, 4, 1, 2, 3) # (B, H, W, D, C) -> (B, C, H, W, D)
        if self.training:
            y = self.conv3x3x3(x)     + \
                self.conv1x1x1_3x3x3(x) + \
                self.conv1x1x1_sbx(x) + \
                self.conv1x1x1_sby(x) + \
                self.conv1x1x1_sbz(x) + \
                self.conv1x1x1_lpl(x)
            if self.with_idt:
                y += x
        else:
            RK, RB = self.rep_params()
            y = F.conv3d(input=x, weight=RK, bias=RB, stride=1, padding=1)
        
        y = self.conv3x3x3(x)     + \
            self.conv1x1x1_3x3x3(x) + \
            self.conv1x1x1_sbx(x) + \
            self.conv1x1x1_sby(x) + \
            self.conv1x1x1_sbz(x) + \
            self.conv1x1x1_lpl(x)
        if self.with_idt:
            y += x
        
        if self.permute:
            y = y.permute(0, 2, 3, 4, 1) # [B, C, H, W, D] --> [B, H, W, D, C]
            y = self.norm(y)
        
        if self.act_type != 'linear':
            y = y.permute(0, 4, 1, 2, 3) # (B, H, W, D, C) -> (B, C, H, W, D)
            y = self.act(y)
            y = y.permute(0, 2, 3, 4, 1) # [B, C, H, W, D] --> [B, H, W, D, C]
        
        return y

    def rep_params(self):
        K0, B0 = self.conv3x3x3.weight, self.conv3x3x3.bias
        K1, B1 = self.conv1x1x1_3x3x3.rep_params()
        K2, B2 = self.conv1x1x1_sbx.rep_params()
        K3, B3 = self.conv1x1x1_sby.rep_params()
        K4, B4 = self.conv1x1x1_sbz.rep_params()
        K5, B5 = self.conv1x1x1_lpl.rep_params()
        RK, RB = (K0+K1+K2+K3+K4+K5), (B0+B1+B2+B3+B4+B5)

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB

class EoS(nn.Module):
    def __init__(self, dim, depth_multiplier=2, act_type='linear', with_idt=False, permute=True, **kwargs):
        super(EoS, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = dim
        self.out_planes = dim
        self.act_type = act_type
        self.permute = permute
        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv3x3x3 = nn.Conv3d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv1x1x1_sbx = SeqConv3x3x3('conv1x1x1-sobelx', self.inp_planes, self.out_planes, -1)
        self.conv1x1x1_sby = SeqConv3x3x3('conv1x1x1-sobely', self.inp_planes, self.out_planes, -1)
        self.conv1x1x1_sbz = SeqConv3x3x3('conv1x1x1-sobelz', self.inp_planes, self.out_planes, -1)
        self.norm = nn.LayerNorm(normalized_shape=dim, elementwise_affine=False)
        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.permute:
            x = x.permute(0, 4, 1, 2, 3) # (B, H, W, D, C) -> (B, C, H, W, D)
        
        if self.training:
            y = self.conv3x3x3(x)     + \
                self.conv1x1x1_sbx(x) + \
                self.conv1x1x1_sby(x) + \
                self.conv1x1x1_sbz(x)
            if self.with_idt:
                y += x
        else:
            RK, RB = self.rep_params()
            y = F.conv3d(input=x, weight=RK, bias=RB, stride=1, padding=1)
        
        # y = self.conv3x3x3(x)     + \
        #     self.conv1x1x1_sbx(x) + \
        #     self.conv1x1x1_sby(x) + \
        #     self.conv1x1x1_sbz(x)
        # if self.with_idt:
        #     y += x

        if self.permute: 
            y = y.permute(0, 2, 3, 4, 1) # [B, C, H, W, D] --> [B, H, W, D, C]
            y = self.norm(y)
        
        if self.act_type != 'linear':
            y = y.permute(0, 4, 1, 2, 3) # (B, H, W, D, C) -> (B, C, H, W, D)
            y = self.act(y)
            y = y.permute(0, 2, 3, 4, 1) # [B, C, H, W, D] --> [B, H, W, D, C]
        return y
    
    def rep_params(self):
        K0, B0 = self.conv3x3x3.weight, self.conv3x3x3.bias
        K1, B1 = self.conv1x1x1_sbx.rep_params()
        K2, B2 = self.conv1x1x1_sby.rep_params()
        K3, B3 = self.conv1x1x1_sbz.rep_params()
        RK, RB = (K0+K1+K2+K3), (B0+B1+B2+B3)

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB


class EoL(nn.Module):
    def __init__(self, dim, depth_multiplier=2, act_type='linear', with_idt=False, permute=True, **kwargs):
        super(EoL, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = dim
        self.out_planes = dim
        self.act_type = act_type
        self.permute = permute
        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False
    
        self.conv3x3x3 = nn.Conv3d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv1x1x1_lpl = SeqConv3x3x3('conv1x1x1-laplacian', self.inp_planes, self.out_planes, -1)
        self.norm = nn.LayerNorm(normalized_shape=dim, elementwise_affine=False)
        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.permute:
            x = x.permute(0, 4, 1, 2, 3) # (B, H, W, D, C) -> (B, C, H, W, D)
        
        if self.training:
            y = self.conv3x3x3(x)     + \
                self.conv1x1x1_lpl(x)
            if self.with_idt:
                y += x
        else:
            RK, RB = self.rep_params()
            y = F.conv3d(input=x, weight=RK, bias=RB, stride=1, padding=1)
            
        # y = self.conv3x3x3(x)     + \
        #     self.conv1x1x1_lpl(x)
        # if self.with_idt:
        #     y += x
        
        if self.permute:
            y = y.permute(0, 2, 3, 4, 1) # [B, C, H, W, D] --> [B, H, W, D, C]
            y = self.norm(y)
            
        if self.act_type != 'linear':
            y = y.permute(0, 4, 1, 2, 3) # (B, H, W, D, C) -> (B, C, H, W, D)
            y = self.act(y)
            y = y.permute(0, 2, 3, 4, 1) # [B, C, H, W, D] --> [B, H, W, D, C]
        return y
    
    def rep_params(self):
        K0, B0 = self.conv3x3x3.weight, self.conv3x3x3.bias
        K1, B1 = self.conv1x1x1_lpl.rep_params()
        RK, RB = (K0+K1), (B0+B1)

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB
    