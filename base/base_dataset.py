import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy import ndimage
from utils import sobel


class BaseDataSet(Dataset):
    def __init__(self, root, split, mean, std, base_size=None, augment=True, val=False,
                 crop_size=321, scale=False, flip=True, rotate=False, blur=False, return_id=False, idx_remove=[],
                 new_files=None, ratio=.1, unlabelled=False, edges=False):
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size
        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur
        self.val = val
        self.files = []
        self.idx = self._set_files(idx_remove=idx_remove, new_files=new_files, ratio=ratio, unlabelled=unlabelled)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.return_id = return_id
        self.unlabelled = unlabelled
        self.edges = edges
        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError

    def _load_data(self, index):
        raise NotImplementedError

    def _val_augmentation(self, image, label):

        h, w, _ = image.shape
        # Scaling, we set the bigger to base size, and the smaller
        # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing
        if self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
            else:
                longside = self.base_size
            # h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
            h, w = (longside, longside // 2) if h > w else (longside // 2, longside)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        if self.crop_size:
            h, w = label.shape
            # Scale the smaller side to crop size
            if h < w:
                h, w = (self.crop_size, int(self.crop_size * w / h))
            else:
                h, w = (int(self.crop_size * h / w), self.crop_size)

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.int32)

            # Center Crop
            h, w = label.shape
            start_h = (h - self.crop_size) // 2
            start_w = (w - self.crop_size) // 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]
        return image, label

    def _unlabeled_augmentation(self, image):

        h, w, _ = image.shape
        # Scaling, we set the bigger to base size, and the smaller
        # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing
        if self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
            else:
                longside = self.base_size
            # h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)

            h, w = (longside, longside // 2) if h > w else (longside // 2, longside)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.crop_size:
            h, w = image.shape
            # Scale the smaller side to crop size
            if h < w:
                h, w = (self.crop_size, int(self.crop_size * w / h))
            else:
                h, w = (int(self.crop_size * h / w), self.crop_size)

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            # Center Crop
            h, w = image.shape
            start_h = (h - self.crop_size) // 2
            start_w = (w - self.crop_size) // 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
        return image

    def _augmentation(self, image, label):
        h, w, _ = image.shape
        # Scaling, we set the bigger to base size, and the smaller
        # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing
        if self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
            else:
                longside = self.base_size
            # h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
            h, w = (longside, longside // 2) if h > w else (longside // 2, longside)
            if self.base_size==2048 and label.shape[1]==2048:
                # image=cv2.resize(image,(512,256),interpolation=cv2.INTER_LINEAR)
                label=cv2.resize(label,(512,256),interpolation=cv2.INTER_NEAREST)
                label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            elif self.base_size==label.shape[1]:
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
                # label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            elif self.base_size==image.shape[1]:
                label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        h, w, _ = image.shape
        # Rotate the image with an angle between -10 and 10
        if self.rotate:
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR , borderMode=cv2.BORDER_REFLECT)
            label = cv2.warpAffine(label, rot_matrix, (w, h),
                                   flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT)

        # Padding to return the correct crop size
        if self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT, }
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
                label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)

            # Cropping
            h, w, _ = image.shape
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        # Random H flip
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()

        # Gaussian Blud (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
                                     borderType=cv2.BORDER_REFLECT_101)
        return image, label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.unlabelled:
            image, image_id = self._load_data(index)
            if self.augment:
                image=self._unlabeled_augmentation(image)
            image = np.uint8(image)
            edges = sobel.main(image)
            image = Image.fromarray(image)
            edges = torch.from_numpy(edges)
            return self.normalize(self.to_tensor(image)), edges, image_id, index
        image, label, image_id = self._load_data(index)
        if self.val:
            if self.augment:
                image, label = self._val_augmentation(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        if self.edges:
            edges = sobel.main(image)
            edges = torch.from_numpy(edges).long()
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))

        if self.edges:
            if self.return_id:
                return self.normalize(self.to_tensor(image)), label, edges, image_id, index
            return self.normalize(self.to_tensor(image)), label, edges
        if self.return_id:
            return self.normalize(self.to_tensor(image)), label, image_id, index
        return self.normalize(self.to_tensor(image)), label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

