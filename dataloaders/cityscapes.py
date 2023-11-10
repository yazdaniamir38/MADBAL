from base import BaseDataSet, BaseDataLoader
from utils import palette
from glob import glob
import numpy as np
import os
from PIL import Image

import random
random.seed(0)
ignore_label = 255
ID_TO_TRAINID = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                    3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                    7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                    14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                    18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                    28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

class CityScapesDataset(BaseDataSet):
    def __init__(self, mode='fine', **kwargs):
        self.num_classes = 19
        self.mode = mode
        self.palette = palette.CityScpates_palette
        self.id_to_trainId = ID_TO_TRAINID
        super(CityScapesDataset, self).__init__(**kwargs)

    def _set_files(self,ratio=0.1,new_files=None,idx_remove=[],unlabelled=False):
        if new_files:
            self.files.extend(new_files)
        elif unlabelled:
            self.files=glob(os.path.join(self.root,'*.pkl'))
        elif self.split=='val':
            self.files=[]
            for city in os.listdir(self.root):
                self.files.extend(sorted(glob(os.path.join(self.root, city, '*labelIds.png'))))
        else:
            self.files=glob(os.path.join(self.root,'*.png'))

    def _load_data(self, index):
        if self.files[index][-3:]=='pkl':
            image_id=os.path.split(self.files[index])[1][:-4]
            image_path=os.path.join('../data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train',image_id.split('_')[0],image_id)
            image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
            return image,image_id

        label_path = self.files[index]
        if not self.val:
            label_path=os.path.join('/cvdata/amir/AL/data/cityscapes/gtFine_trainvaltest/gtFine/train',os.path.split(label_path)[1].split('_')[0],os.path.split(label_path)[1])
        image_id=os.path.split(label_path)[1][:-19]+'leftImg8bit.png'
        image_path=os.path.join('../data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit',self.split,image_id.split('_')[0],image_id)
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        for k, v in self.id_to_trainId.items():
            label[label == k] = v


        return image, label, image_id




class CityScapes(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, mode='fine', val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False,idx_remove=[],new_files=None,ratio=.1,unlabelled=False,edges=False):

        self.MEAN = [0.28689529, 0.32513294, 0.28389176]
        self.STD = [0.17613647, 0.18099176, 0.17772235]

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
            'idx_remove':idx_remove,
            'new_files':new_files,
            'ratio':ratio,
            'unlabelled':unlabelled,
            'edges':edges
        }
        self.dataset = CityScapesDataset(mode=mode, **kwargs)
        super(CityScapes_super_pixels, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

