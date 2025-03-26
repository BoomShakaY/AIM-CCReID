import torchvision
from torch import nn
from torch.nn import init
from models.utils import pooling
        

class ResNet50(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        # or passing weights=ResNet50_Weights.IMAGENET1K_V1  instead of pretrained=True to haddle the warning
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride=(1, 1)
            resnet50.layer4[0].downsample[0].stride=(1, 1) 
        # self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool

        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))

        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        
    def forward(self, tmp):
        tmp = self.conv1(tmp)
        tmp = self.bn1(tmp)
        tmp = self.relu(tmp)
        tmp = self.maxpool(tmp)

        tmp = self.layer1(tmp)
        tmp = self.layer2(tmp)
        tmp = self.layer3(tmp)
        old_x = self.layer4(tmp) # torch.Size([32, 2048, 24, 12])
        
        # old_x = self.base(tmp)

        x = self.globalpooling(old_x) # torch.Size([32, 4096, 1, 1])
        x = x.view(x.size(0), -1)
        f = self.bn(x)

        return old_x, f