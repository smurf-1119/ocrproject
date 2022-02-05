from ast import iter_child_nodes
from turtle import forward
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Callable, Union, List, Optional

class CReLU(nn.Module):
    def __init__(self) -> None:
        super(CReLU,self).__init__()
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        output = torch.cat((self.relu(x),self.relu(-x)),1)
        return output
 
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = CReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups 
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width//2)
        self.bn1 = norm_layer(width//2)
        self.conv2 = conv3x3(width, width//2, stride, groups, dilation)
        self.bn2 = norm_layer(width//2)
        # self.conv3 = nn.Sequential()
        self.conv3 = conv1x1(width, planes * self.expansion //2)
        self.bn3 = norm_layer(planes * self.expansion // 2)
        # self.conv3_downsample = nn.Sequential(
        #     conv1x1(width, planes * self.expansion//2),
        #     norm_layer(planes * self.expansion//2)
        # )



        self.crelu = CReLU()
        self.relu = nn.ReLU()
        # self.downsample = downsample
        if downsample == None:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion // 2, stride = 1),
                norm_layer(planes * self.expansion // 2),
            ) #downsample 本质是一个1*1的卷积，expansion为4,下采样成四分之一
        else:
            self.downsample = downsample

        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.crelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.crelu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.crelu(out) #改动

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.inplanes_1 = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes_1, kernel_size=7, stride=2, padding=3,
                               bias=False) #[3,224,224]输入(C,H,W)
        self.bn1 = norm_layer(self.inplanes_1)
        self.relu = CReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #stage 0的所有描述，输出为[64,56,56]
        self.layer1 = self._make_layer(block, 64, layers[0])#Stage 1 输入为[64,56,56],layer=3,有3层
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer #确定归一化层
        # downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         conv1x1(self.inplanes, planes * block.expansion, stride),
        #         norm_layer(planes * block.expansion),
        #     ) #downsample 本质是一个1*1的卷积，expansion为4,下采样成四分之一
        
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes * block.expansion // 2, stride),
            norm_layer(planes * block.expansion // 2),
        ) #downsample 本质是一个1*1的卷积，expansion为4,下采样成四分之一

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)#2048，7，7
        # x = torch.flatten(x, 1)#不需要这个输出
        # x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    pretrained_path:str
) -> ResNet:
    model = ResNet(block, layers)
    if pretrained:
        state_dict = torch.load(pretrained_path)
        # current_state_dict = model.state_dict()
        # print(current_state_dict.keys())
        # new_state_dict = {key:value for key,value in state_dict.items() if key in current_state_dict.keys()}
        # current_state_dict.update(new_state_dict)
        # torch.save(current_state_dict,'./resnet50_new.pth')
        # print('finish')
        model.load_state_dict(state_dict)
    return model

def resnet50(pretrained: bool = False,pretrained_path=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], pretrained,pretrained_path)


pretrained_path = "./resnet50_new.pth"
model = resnet50(pretrained=False,pretrained_path=pretrained_path)
input_images = torch.zeros((1,3,224,224))
out = model(input_images)
print(out.size())