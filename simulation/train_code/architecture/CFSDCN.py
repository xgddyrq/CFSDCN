import math


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from .ops_dcnv3 import modules as opsm


class FeedForward(nn.Module):
    def __init__(self, dim, cnt=0, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult,dim * mult,kernel_size=3,padding=1,stride=1,groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )
    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Embed(nn.Module):
    def __init__(self,
                 in_chans=28,
                 embed_dim=28,
                 ):
        super().__init__()

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        
        x = self.norm(x)

        return x

class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)



def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


class DCNLayer(nn.Module):
    def __init__(self,
                 core_op,
                 channels,
                 groups,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 offset_scale=1.0,
                 dw_kernel_size=None,
                 center_feature_scale=False,
                 remove_center=False,
                 ):
        super().__init__()
        self.channels = channels
        self.groups = groups

        self.norm1 = build_norm_layer(channels, 'LN')
        self.dcn = core_op(
            channels=channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=groups,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size,
            center_feature_scale=center_feature_scale,
            remove_center=remove_center,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        self.ca = FeedForward(dim=channels)

    def forward(self, x):
        x = x + self.drop_path(self.dcn(self.norm1(x)))
        x = x + self.drop_path(self.ca(self.norm2(x)))
        return x

class DCNBlock(nn.Module):
    def __init__(self,
                 core_op,
                 channels,
                 groups,
                 num_blocks = 2,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 offset_scale=1.0,
                ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.cnt = 0
        for _ in range(num_blocks):
            self.blocks.append(
                DCNLayer(core_op=core_op,
                         channels=channels,
                         groups=groups,
                         drop_path=drop_path,
                         act_layer=act_layer,
                         norm_layer=norm_layer,
                         offset_scale=offset_scale,
                        ),
        )

    def forward(self, x):
        for dcn in self.blocks:
            x = dcn(x)
        return x

class CFSDCN(nn.Module):
    def __init__(self, dim=28, core_op='DCNv3', groups=14, embed_dim=28, stage=2, num_blocks=[2, 2, 2]):
        super(CFSDCN, self).__init__()
        self.dim = dim
        self.stage = stage
        self.embed_dim = embed_dim
        self.fution = nn.Conv2d(56, 28, 1, 1, 0, bias=False)
        self.patch_embed = Embed(in_chans=dim,
                                 embed_dim=embed_dim
                                )
        self.encoder_layers = nn.ModuleList([])
        dim_stage = embed_dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                DCNBlock(core_op=getattr(opsm, core_op),
                         channels=dim_stage,
                         groups=groups,
                         num_blocks=num_blocks[i]
                         ),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        self.bottleneck = DCNBlock(core_op=getattr(opsm, core_op),
                                   channels=dim_stage,
                                   groups=groups,
                                   num_blocks=num_blocks[-1],
                                   )

        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                DCNBlock(core_op=getattr(opsm, core_op),
                         channels=dim_stage // 2,
                         groups=groups,
                         num_blocks=num_blocks[stage - 1 - i],
                         ),
            ]))
            dim_stage //= 2

        self.mapping = nn.Conv2d(self.dim, 28, 3, 1, 1, bias=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((1, 28, 256, 256)).to(x.device)
        x = self.fution(torch.cat([x, mask], dim=1))
        fea = self.lrelu(self.patch_embed(x))
        # Encoder
        fea_encoder = []
        b, n, c = fea.shape
        h = int(math.sqrt(n))
        w = int(math.sqrt(n))
        fea = fea.view(b, h, w, c)
        for i,(DCNLayer, FeaDownSample) in enumerate(self.encoder_layers):
            fea = DCNLayer(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea.permute(0,3,1,2)).permute(0,2,3,1)
           
        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea.permute(0,3,1,2)).permute(0,2,3,1)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=3).permute(0,3,1,2))
            fea = LeWinBlcok(fea.permute(0,2,3,1))

        # Mapping
        out = self.mapping(fea.permute(0,3,1,2)) + x

        return out