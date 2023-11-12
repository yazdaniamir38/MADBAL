from base import BaseModel
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
from utils.helpers import initialize_weights,set_trainable
from itertools import chain
import copy
from collections import OrderedDict
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, dropRate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.conv2(x1)
        out = x + x2
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out

class mobile_net_lossy_sequential(BaseModel):
    def __init__(self, num_classes=19,backbone="",freeze_backbone=False,freeze_bn=False):
        super(mobile_net_lossy_sequential, self).__init__()
        mobilenet=models.segmentation.lraspp_mobilenet_v3_large(pretrained= True, progress= False, num_classes= 21, pretrained_backbone= True)
        self.backbone=mobilenet.backbone
        self.backbone.return_layers.update({'6':'mid1','11':'mid2','14':'mid3'})
        self.main_classifier=mobilenet.classifier
        self.main_classifier.high_classifier=nn.Conv2d(128,num_classes,1)
        self.main_classifier.low_classifier = nn.Conv2d(40, num_classes, 1)
        self.W_loss_pred=copy.deepcopy(self.main_classifier)
        self.W_loss_pred.high_classifier=nn.Conv2d(128,3,1)
        self.W_loss_pred.low_classifier=nn.Conv2d(40,3,1)
        self.loss_pred_conv=nn.Conv2d(960,480,kernel_size=3,padding=1,stride=1)
        self.loss_pred = nn.Sequential(nn.Conv2d(539, 256, kernel_size=3, stride=1, padding=1),
                                       nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1), ResidualBlock(64),
                                       SELayer(64),
                                       nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1))
        self.middle_conv1=nn.Conv2d(40,num_classes,kernel_size=3,stride=1,padding=1)
        # self.middle_conv2 = nn.Conv2d(112, num_classes, kernel_size=3, stride=1, padding=1)
        self.middle_deconv2=nn.ConvTranspose2d(112,num_classes,kernel_size=5,stride=2,padding=2,output_padding=1)
        self.middle_deconv3=nn.ConvTranspose2d(160,num_classes,kernel_size=5,stride=2,padding=2,output_padding=1)
    def forward(self,x,stage=-1):
        features=self.backbone(x)
        middle_1,middle_2,middle_3=features['mid1'],features['mid2'],features['mid3']
        features=OrderedDict({'low':features['low'],'high':features['high']})
        main_seg=self.main_classifier(features)
        main_seg = F.interpolate(main_seg, size=x.shape[-2:], mode='bilinear', align_corners=False)
        if stage==1:
            return main_seg

        W = F.softmax(self.W_loss_pred(features))
        features['high'] = F.interpolate(
            self.loss_pred_conv(features['high']), size=features['low'].shape[-2:],
            mode='bilinear', align_corners=False)
        middle_1 = self.middle_conv1(middle_1)
        middle_2 = self.middle_deconv2(middle_2)
        middle_3 = self.middle_deconv3(middle_3)
        loss = F.sigmoid(
            F.interpolate(self.loss_pred(torch.cat(
                [W[:, 0:1, :, :] * middle_1 + W[:, 1:2, :, :] * middle_2 + W[:, 2:3, :, :] * middle_3,
                 features['low'],
                 features['high']], dim=1)), size=x.shape[-2:],
                mode='bilinear'))
        # loss = F.sigmoid(
        #     F.interpolate(self.loss_pred(torch.cat(
        #         [features['low'],
        #          features['high']], dim=1)), size=x.shape[-2:],
        #         mode='bilinear'))
        middle_1 = F.interpolate(middle_1, size=x.shape[-2:], mode='bilinear')
        middle_2 = F.interpolate(middle_2, size=x.shape[-2:], mode='bilinear')
        middle_3 = F.interpolate(middle_3, size=x.shape[-2:], mode='bilinear')
        if stage==2:
            return main_seg,loss,middle_1,middle_2,middle_3
        else:
            W = F.interpolate(W, size=x.shape[-2:], mode='bilinear')
            # return main_seg, loss
            return main_seg,middle_1,middle_2,middle_3,W,loss


    def get_non_lossy_parameters(self):
        return chain(self.backbone.parameters(), self.main_classifier.parameters())

    def get_lossy_parameters(self):
        return chain(self.loss_pred.parameters(),self.loss_pred_conv.parameters(),self.W_loss_pred.parameters(),self.middle_conv1.parameters(),self.middle_deconv2.parameters(),self.middle_deconv3.parameters())

    def freeze_everything_but_lossy_Detector(self,state=False):

        for name,module in self.named_modules():
            if 'loss_pred' not in name:
                for params in module.parameters():
                    params.requires_grad=state
                if isinstance(module, nn.BatchNorm2d):
                    if state:
                        module.train()
                    else:
                        module.eval()
            else:
                for params in module.parameters():
                    params.requires_grad=not state
                if isinstance(module, nn.BatchNorm2d):
                    if not state:
                        module.train()
                    else:
                        module.eval()
            if 'middle' in name:
                for params in module.parameters():
                    params.requires_grad=True
                if isinstance(module, nn.BatchNorm2d):
                    module.train()

# model=mobile_net_lossy_sequential()
# x=torch.zeros([1,3,1024,2048])
# out=model(x)
# print('done')