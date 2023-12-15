import torch
import torch.nn as nn
import torch.nn.functional as F

from .invpt import InvPT
import pdb
import numpy as np
from einops import rearrange as o_rearrange
INTERPOLATE_MODE = 'bilinear'
BATCHNORM = nn.SyncBatchNorm # nn.BatchNorm2d

def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerDecoder(nn.Module):

    def __init__(self, p):
        super().__init__()

        self.embed_dim = p.embed_dim
        
        embed_dim_with_pred = self.embed_dim + p.PRED_OUT_NUM_CONSTANT
        p.mtt_resolution = [_ // p.mtt_resolution_downsample_rate for _ in p.spatial_dim[-1]] # resolution at the input of transformer decoder
        self.p = p

        spec = {
            'ori_embed_dim': self.embed_dim,
            'NUM_STAGES': 3,
            'PATCH_SIZE': [0, 3, 3],
            'PATCH_STRIDE': [0, 1, 1],
            'PATCH_PADDING': [0, 2, 2],
            'DIM_EMBED': [embed_dim_with_pred, embed_dim_with_pred//2, embed_dim_with_pred//4],
            'NUM_HEADS': [2, 2, 2],
            'MLP_RATIO': [4., 4., 4.],
            'DROP_PATH_RATE': [0.15, 0.15, 0.15],
            'QKV_BIAS': [True, True, True],
            'KV_PROJ_METHOD': ['avg', 'avg', 'avg'],
            'KERNEL_KV': [2, 4, 8],
            'PADDING_KV': [0, 0, 0],
            'STRIDE_KV': [2, 4, 8],
            'Q_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
            'KERNEL_Q': [3, 3, 3],
            'PADDING_Q': [1, 1, 1],
            'STRIDE_Q': [2, 2, 2],
        }

        # intermediate supervision
        input_channels = p.backbone_channels[-1]
        task_channels  = self.embed_dim
        self.intermediate_head = nn.ModuleDict()
        self.invpt = InvPT(p, in_chans=embed_dim_with_pred, spec=spec)
        self.preliminary_decoder = nn.ModuleDict()

        for t in p.TASKS.NAMES:
            self.intermediate_head[t] = nn.Conv2d(task_channels, p.TASKS.NUM_OUTPUT[t], 1) 
            self.preliminary_decoder[t] = nn.Sequential(
                                            ConvBlock(input_channels, input_channels),
                                            ConvBlock(input_channels, task_channels),
                                        )

        self.scale_embed = nn.ModuleList()
        self.scale_embed.append(nn.ConvTranspose2d(p.backbone_channels[0], spec['DIM_EMBED'][2], kernel_size=3, stride=2, padding=1,output_padding=1))
        self.scale_embed.append(nn.Conv2d(p.backbone_channels[1], spec['DIM_EMBED'][1], 3, padding=1))
        self.scale_embed.append(nn.Conv2d(p.backbone_channels[2], spec['DIM_EMBED'][0], 3, padding=1))
        self.scale_embed.append(None)


    def forward(self, x_list):
        '''
        Input:
        Backbone multi-scale feature list: 4 * x: tensor [B, embed_dim, h, w]
        '''

        # intermediate supervision
        ms_feat_dict = {}
        inter_pred = {}
        back_fealist = {}
        #back_fea = []
        
        for task in self.p.TASKS.NAMES:
            back_fea = []
            for sca in range(len(x_list[task])):
                oh, ow = self.p.spatial_dim[sca]
                _fea = x_list[task][sca]
                _fea = rearrange(_fea, 'b (h w) c -> b c h w', h=oh, w=ow)
                if sca == 3:
                    x = _fea # use last scale feature as input of InvPT decoder
                if self.scale_embed[sca] != None:
                    _fea = self.scale_embed[sca](_fea)

                back_fea.append(_fea)#

            back_fealist[task] = back_fea
            h, w = self.p.mtt_resolution
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            _x = self.preliminary_decoder[task](x)
            ms_feat_dict[task] = _x#

            _inter_p = self.intermediate_head[task](_x)
            inter_pred[task] = _inter_p
        #pdb.set_trace()
        if len(self.p.TASKS.NAMES) == 4: #nyud
            back_fea = [ (i+j+k+l)/(len(self.p.TASKS.NAMES)) for i, j, k ,l in zip(back_fealist[self.p.TASKS.NAMES[0]], back_fealist[self.p.TASKS.NAMES[1]], 
                                                        back_fealist[self.p.TASKS.NAMES[2]],back_fealist[self.p.TASKS.NAMES[3]])]
        elif len(self.p.TASKS.NAMES) == 5: #pascal
            back_fea = [ (i+j+k+l+m)/(len(self.p.TASKS.NAMES)) for i, j, k , l, m in zip(back_fealist[self.p.TASKS.NAMES[0]], back_fealist[self.p.TASKS.NAMES[1]], 
                                                        back_fealist[self.p.TASKS.NAMES[2]],back_fealist[self.p.TASKS.NAMES[3]], back_fealist[self.p.TASKS.NAMES[4]])]
        else:
            raise NotImplementedError
        
        x_dict = self.invpt(ms_feat_dict, inter_pred, back_fea) # multi-scale input
        return x_dict, inter_pred

class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BATCHNORM
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out

class MLPHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MLPHead, self).__init__()

        self.linear_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.linear_pred(x) 