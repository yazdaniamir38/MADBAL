# Originally written by Kazuto Nakashima
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob

class VOCDataset(BaseDataSet):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """

    def __init__(self, **kwargs):
        self.num_classes = 21
        self.palette = palette.get_voc_palette(self.num_classes)
        super(VOCDataset, self).__init__(**kwargs)

    def _set_files(self,idx_remove=[], new_files=None, ratio=0.1, unlabelled=False):
        if unlabelled:
            self.files=glob(os.path.join(self.root,'*.pkl'))
        elif self.split=='val':
            self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')
            file_list = os.path.join(self.root, "ImageSets/Segmentation", self.split + ".txt")
            self.files = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]

        else:
            self.files = glob(os.path.join(self.root, '*.png'))

    def _load_data(self, index):
        if self.files[index][-3:]=='pkl':
            # index=16
            image_id = os.path.split(self.files[index])[1][:-3]+'jpg'
            image_path = os.path.join('../data/VOCdevkit/VOC2012/JPEGImages', image_id)
            image = np.asarray(Image.open(image_path), dtype=np.float32)
            return image,image_id
        if self.split=='val':
            image=np.asarray(Image.open(os.path.join(self.root,'JPEGImages',self.files[index][0]+'.jpg')), dtype=np.float32)
            label=np.asarray(Image.open(os.path.join('../data/VOCdevkit/VOC2012/SegmentationClass',self.files[index][0]+'.png')), dtype=np.int32)
            image_id=os.path.split(self.files[index][0])[1]
            # label[label==0]=255
            # label[label!=255]-=1
            return image,label,image_id

        image_id = os.path.split(self.files[index])[1][:-3]+'jpg'
        image_path=os.path.join('../data/VOCdevkit/VOC2012/JPEGImages/',image_id)
        label_path=self.files[index]
        # label_path=os.path.join('../data/VOCdevkit/VOC2012/SegmentationClass',image_id[:-3]+'png')
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        # label[label==0]=255
        # label[label!=255]-=1
        return image, label, image_id


class VOCAugDataset(BaseDataSet):
    """
    Contrains both SBD and VOC 2012 dataset
    Annotations : https://github.com/DrSleep/tensorflow-deeplab-resnet#evaluation
    Image Sets: https://ucla.app.box.com/s/rd9z2xvwsfpksi7mi08i2xqrj7ab4keb/file/55053033642
    """

    def __init__(self, **kwargs):
        self.num_classes = 21
        self.palette = palette.get_voc_palette(self.num_classes)
        super(VOCAugDataset, self).__init__(**kwargs)

    def _set_files(self,idx_remove=[], new_files=None, ratio=0.1, unlabelled=False):
        if unlabelled:
            self.files=glob(os.path.join(self.root,'*.pkl'))
        elif self.split=='val':
            self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')
            file_list = os.path.join(self.root, "ImageSets/Segmentation", self.split + ".txt")
            self.files = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]

        else:
            self.files = glob(os.path.join(self.root, '*.png'))

    def _load_data(self, index):
        if self.files[index][-3:]=='pkl':
            index=16
            image_id = os.path.split(self.files[index])[1][:-4]
            image_path = os.path.join('../data/VOCdevkit/VOC2012/JPEGImages', image_id)
            image = np.asarray(Image.open(image_path), dtype=np.float32)
            return image,image_id
        if self.split=='val':
            image=np.asarray(Image.open(self.files[index]), dtype=np.float32)
            label=np.asarray(Image.open(os.path.join('../data/VOCdevkit/VOC2012/SegmentationClass',self.files[index])), dtype=np.int32)
            image_id=os.path.split(self.files[index])[1]
            return image,label,image_id

        image_id = os.path.split(self.files[index])[1]
        image_path=os.path.join('../data/VOC_devkit/VOC2012/JPEGImages/',image_id)
        # label_path=os.path.join('../data/VOC_devkit/VOC2012/SegmentationClass',image_id)
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(self.files[index], dtype=np.int32)
        return image, label, image_id


class VOC(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=False, num_workers=1,
                 val=False,
                 shuffle=False, flip=False, rotate=False, blur=False, augment=False, val_split=None, return_id=False,idx_remove=[],new_files=None,ratio=.1,unlabelled=False,edges=False):

        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        #self.MEAN=[0.485, 0.456, 0.406]
        self.STD = [0.23965294, 0.23532275, 0.2398498]
       # self.STD= [0.229, 0.224, 0.225]
        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val,
            'idx_remove': idx_remove,
            'new_files': new_files,
            'ratio': ratio,
            'unlabelled': unlabelled,
            'edges': edges
        }

        if split in ["train_aug", "trainval_aug", "val_aug", "test_aug"]:
            self.dataset = VOCAugDataset(**kwargs)
        elif split in ["train", "trainval", "val", "test"]:
            self.dataset = VOCDataset(**kwargs)
        else:
            raise ValueError(f"Invalid split name {split}")
        super(VOC_superpixels, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

