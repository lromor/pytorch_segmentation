from base import BaseModel
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
from utils.helpers import initialize_weights
from itertools import chain
import ebconv

from .deeplabv3_plus import ResNet, Xception, Decoder, assp_branch

'''
-> The Atrous Spatial Pyramid Pooling
'''

def assp_branch_cbsconv(in_channels, out_channels, kernel_size, dilation, layout, groups, scaling):
    padding = 0 if kernel_size == 1 else dilation
    new_kernel_size = kernel_size * dilation - dilation + 1
    init_region = (kernel_size, kernel_size)
    conv = ebconv.nn.CBSConv2d(
        in_channels, out_channels, new_kernel_size,
        padding=padding, nc=9, k=2, layout=layout, basis_groups=groups,
        adaptive_centers=True, adaptive_scalings=True, init_region=init_region)

    with torch.no_grad():
        for name, param in conv.named_parameters():
            if name.endswith('centers'):
                param.data *= new_kernel_size / kernel_size
            elif name.endswith('scalings'):
                param.data = torch.ones_like(param)
                volume = new_kernel_size * new_kernel_size
                param.data *= (volume / conv.nc) ** (1 / conv.dims)
                param.data *= scaling

    return nn.Sequential(
        conv,
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))


class ASSP(nn.Module):
    def __init__(self, in_channels, output_stride, layout, groups, scaling):
        super(ASSP, self).__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16: dilations = [1, 6, 12, 18]
        elif output_stride == 8: dilations = [1, 12, 24, 36]
        
        self.aspp1 = assp_branch(in_channels, 256, 1, dilation=dilations[0])
        self.aspp2 = assp_branch_cbsconv(in_channels, 256, 3, dilation=dilations[1], layout=layout, groups=groups, scaling=scaling)
        self.aspp3 = assp_branch_cbsconv(in_channels, 256, 3, dilation=dilations[2], layout=layout, groups=groups, scaling=scaling)
        self.aspp4 = assp_branch_cbsconv(in_channels, 256, 3, dilation=dilations[3], layout=layout, groups=groups, scaling=scaling)

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        
        self.conv1 = nn.Conv2d(256*5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x

'''
-> Deeplab V3 + CBSConv
'''
def filter_bases_params(data):
    name, _ = data
    if name.endswith('centers') or name.endswith('scalings'):
        return True
    return False

class DeepLabCBSConv(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='xception', pretrained=True, 
                 output_stride=16, freeze_bn=False, layout='grid', groups=4,
                 scaling=1.5, **_):
                
        super(DeepLabCBSConv, self).__init__()
        assert ('xception' or 'resnet' in backbone)
        if 'resnet' in backbone:
            self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained)
            low_level_channels = 256
        else:
            self.backbone = Xception(output_stride=output_stride, pretrained=pretrained)
            low_level_channels = 128

        self.ASSP = ASSP(in_channels=2048, output_stride=output_stride, layout=layout,
                         groups=groups, scaling=scaling)
        self.decoder = Decoder(low_level_channels, num_classes)

        if freeze_bn: self.freeze_bn()

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        x = self.ASSP(x)
        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

    # Two functions to yield the parameters of the backbone
    # & Decoder / ASSP to use differentiable learning rates
    # FIXME: in xception, we use the parameters from xception and not aligned xception
    # better to have higher lr for this backbone

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        ASPP_params = []
        for name, param in self.ASSP.named_parameters():
            if filter_bases_params((name, param)):
                continue
            ASPP_params.append(param)
        params = chain(ASPP_params, self.decoder.parameters())
        return params

    def get_splines_params(self):
        bases_params = []
        for name, param in self.ASSP.named_parameters():
            if not filter_bases_params((name, param)):
                continue
            bases_params.append(param)
        return bases_params

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

