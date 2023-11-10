# import resnet, xception, drn, mobilenet
from .mobilenet import MobileNetV2

def build_backbone(backbone, output_stride, BatchNorm, mc_dropout):
    """
    Args:
        backbone:
        output_stride:
        BatchNorm: 在 backbone 网络全部使用 BN or SyncBN
        mc_dropout:
    """
    # if backbone == 'resnet50':
    #     return resnet.ResNet50(output_stride, BatchNorm)
    # elif backbone == 'resnet101':
    #     return resnet.ResNet101(output_stride, BatchNorm)
    # elif backbone == 'xception':
    #     return xception.AlignedXception(output_stride, BatchNorm)
    # elif backbone == 'drn':
    #     return drn.drn_d_54(BatchNorm)
    if backbone == 'mobilenet':
        return MobileNetV2(output_stride, BatchNorm, mc_dropout=mc_dropout)
    else:
        raise NotImplementedError
