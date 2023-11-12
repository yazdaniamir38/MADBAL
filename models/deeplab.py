import torch
import torch.nn as nn
import torch.nn.functional as F
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .backbone import build_backbone
from .aspp import build_aspp
from .decoder import SegmentHead, MaskHead, MaskHead_branch
import os
from .mobilenet_sequential import ResidualBlock,SELayer
from itertools import chain

class DeepLab(nn.Module):

    def __init__(self,
                 num_classes=19,backbone='mobilenet', output_stride=16,
                 sync_bn=False, mc_dropout=False,
                 with_mask=False, with_pam=False, branch_early=False,freeze_bn=False,freeze_backbone=False):
        super(DeepLab, self).__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, mc_dropout)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)

        # low level features
        if backbone.startswith('resnet') or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.low_level_conv = nn.Sequential(nn.Conv2d(low_level_inplanes, 48, 1, bias=False),
                                            BatchNorm(48),
                                            nn.ReLU())
        # segment
        self.seg_head = SegmentHead(num_classes, BatchNorm)
        self.W_loss_pred=SegmentHead(3,BatchNorm)
        self.middle_conv1 = nn.Conv2d(24, num_classes, kernel_size=3, stride=1, padding=1)
        self.mid_deconv2=nn.ConvTranspose2d(64,num_classes,kernel_size=5,stride=4,padding=1,output_padding=1)
        self.mid_deconv3 = nn.ConvTranspose2d(96, num_classes, kernel_size=5, stride=4, padding=1, output_padding=1)
        self.loss_pred = nn.Sequential(nn.Conv2d(256 + num_classes, 512, kernel_size=3, stride=1, padding=1),
                                       nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1), ResidualBlock(64),
                                       SELayer(64),
                                       nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1))

        # error mask -> difficulty branch
        self.with_mask = with_mask
        self.branch_early = branch_early
        if with_mask:
            if branch_early:
                self.mask_head = MaskHead_branch(304, num_classes, BatchNorm, with_pam)
            else:
                self.mask_head = MaskHead(num_classes, with_pam)

        self.return_features = False
        self.return_attention = False

    def forward(self, inputs,stage=-1):
        backbone_feat, mid_1,mid_2,mid_3 = self.backbone(inputs)  # 1/16, 1/4;
        x = self.aspp(backbone_feat)  # 1/16 -> aspp -> 1/16

        # low + high features
        low_level_feat = self.low_level_conv(mid_1)  # 256->48
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)  # 1/4
        second_to_last_features = torch.cat((x, low_level_feat), dim=1)  # 304=256+48

        # segment
        out = self.seg_head(second_to_last_features)
        out = F.interpolate(out, size=inputs.size()[2:], mode='bilinear', align_corners=True)
        if stage==1:
            return out
        mid_1=self.middle_conv1(mid_1)
        mid_2=self.mid_deconv2(mid_2)
        mid_3=self.mid_deconv3(mid_3)
        W=F.softmax(self.W_loss_pred(second_to_last_features),dim=1)
        loss = F.sigmoid(
            F.interpolate(self.loss_pred(torch.cat(
                [W[:, 0:1, :, :] * mid_1 + W[:, 1:2, :, :] * mid_2 + W[:, 2:3, :, :] * mid_3,
                 x], dim=1)), size=inputs.shape[-2:],
                mode='bilinear'))
        mid_1=F.interpolate(mid_1, size=out.shape[-2:], mode='bilinear')
        mid_2 = F.interpolate(mid_2, size=out.shape[-2:], mode='bilinear')
        mid_3 = F.interpolate(mid_3, size=out.shape[-2:], mode='bilinear')
        if stage==2:
            return out,loss,mid_1,mid_2,mid_3
        W = F.interpolate(W, size=out.shape[-2:], mode='bilinear')
        return out, mid_1, mid_2, mid_3, W, loss


    def set_return_features(self, return_features):  # True or False
        self.return_features = return_features

    def set_return_attention(self, return_attention):  # True or False
        self.return_attention = return_attention

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.low_level_conv, self.seg_head]
        if self.with_mask:
            modules.append(self.mask_head)
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def load_pretrain(self, pretrained):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')['state_dict']
            print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}  # 不加载最后的 head 参数
            # for k, v in pretrained_dict.items():
            #     print('=> loading {} | {}'.format(k, v.size()))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            print('No such file {}'.format(pretrained))


    def get_non_lossy_parameters(self):
        return chain(self.backbone.parameters(), self.aspp.parameters(),self.low_level_conv.parameters(),self.seg_head.parameters(),)

    def get_lossy_parameters(self):
        return chain(self.loss_pred.parameters(),self.W_loss_pred.parameters(),self.middle_conv1.parameters(),self.mid_deconv2.parameters(),self.mid_deconv3.parameters())

    def freeze_everything_but_lossy_Detector(self,state=False):

        for name,module in self.named_modules():
            if 'loss_pred' not in name:
                for params in module.parameters():
                    params.requires_grad=state
                if isinstance(module, nn.BatchNorm2d) or isinstance(module,nn.Dropout2d):
                    if state:
                        module.train()
                    else:
                        module.eval()
            else:
                # print(name)
                for params in module.parameters():
                    params.requires_grad=not state
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Dropout2d):
                    if not state:
                        module.train()
                    else:
                        module.eval()
            if 'middle' in name or 'mid' in name:
                for params in module.parameters():
                    params.requires_grad=True
                if isinstance(module, nn.BatchNorm2d):
                    module.train()

