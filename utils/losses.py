import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import class_weight 
from utils.lovasz_losses import lovasz_softmax

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def get_weights(target,n_classes):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    counts=counts[classes!=255]
    classes=classes[classes!=255]
    cls_w = np.median(counts) / counts
    #cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(n_classes)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss

class Binary_CELoss(nn.Module):
    def __init__(self):
        super(Binary_CELoss, self).__init__()
        self.cross_entropy = nn.BCELoss(reduction='none')
        # self.ignore_index = ignore_index
        # self.smooth = smooth
        # self.class_based=class_based
    def forward(self, output, target):
        CE_Loss = self.cross_entropy(output, target)
        mask = target != 255
        CE_Loss = CE_Loss[mask].mean()
        return CE_Loss

        # if self.ignore_index not in range(target.min(), target.max()):
        #     if (target == self.ignore_index).sum() > 0:
        #         target[target == self.ignore_index] = target.min()
        # target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        # output = F.softmax(output, dim=1)
        # if not self.class_based:
        #     output_flat = output.contiguous().view(-1)
        #     target_flat = target.contiguous().view(-1)
        #     intersection = (output_flat * target_flat).sum()
        #     loss = 1 - ((2. * intersection + self.smooth) /
        #                 (output_flat.sum() + target_flat.sum() + self.smooth))
        # else:
        #     intersection=torch.sum(target*output,dim=(2,3))
        #     loss=1-((2. * intersection + self.smooth) /
        #                 (torch.sum(output,dim=(2,3)) + torch.sum(target,dim=(2,3)) + self.smooth))
        # return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()

class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss

# class binary_CELoss(nn.Module):
#     def __int__(self):
#         super(binary_CELoss, self).__init__()
#         self.cross_entropy=nn.BCELoss(reduction=None)
#     def forward(self,output,target):
#         CE_Loss=self.cross_entropy(output,target)
#         mask=target!=255
#         CE_Loss=CE_Loss(mask).mean()
#         return CE_Loss



class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
    
    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss

def mse_loss(input, target, ignored_index, reduction):
        mask = target == ignored_index
        out = (input[~mask]-target[~mask])**2
        if reduction == "mean":
            return out.mean()
        elif reduction == "None":
            return out

def comp_thresh(loss,target,num_classes):
    thresh=torch.zeros(num_classes)
    Counts=torch.zeros(num_classes)
    existing_classes,counts = torch.unique(target,return_counts=True)
    counts=counts[existing_classes!=255]
    existing_classes=existing_classes[existing_classes!=255]
    for j,i in enumerate(existing_classes):
        thresh[i]=loss[target==i].sum()
        Counts[i]=counts[j]
    return thresh,Counts




class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=19,ignore_index=255):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.ignore_index=ignore_index
    def forward(self, inputs, target_oneHot):
        target_oneHot[target_oneHot == self.ignore_index] = target_oneHot.min()
        target_oneHot=make_one_hot(target_oneHot.unsqueeze(1),self.classes)
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.contiguous().view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.contiguous().view(N, self.classes, -1).sum(2)

        loss = inter / (union)

        ## Return average loss over classes and batch
        return -loss.mean()