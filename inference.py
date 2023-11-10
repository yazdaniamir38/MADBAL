import argparse
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import dataloaders
import models
from utils.metrics import eval_metrics, AverageMeter
from collections import OrderedDict


batch_time = AverageMeter()
data_time = AverageMeter()
total_loss = AverageMeter()
total_inter, total_union = 0, 0
total_correct, total_label = 0, 0
# For cityscape only
ignore_label = 255
ID_TO_TRAINID = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                    3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                    7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                    14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                    18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                    28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

def get_instance(module, name, config,*args,idx_remove=False):
    # GET THE CORRESPONDING CLASS / FCT
    if idx_remove:
        return getattr(module, config['type'])(*args,**config['args'])
    else:
        return getattr(module, config[name]['type'])(*args,**config[name]['args'])

def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

    return zeros.scatter(scatter_dim, y_tensor, 1).permute([2,0,1])


def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

def sliding_predict(model, image, normalize,num_classes,flip=False):

    image_size = image.shape
    tile_size=(image_size[2],image_size[3])
    overlap = 0

    stride = (ceil(tile_size[0] * (1 - overlap)),ceil(tile_size[1] * (1 - overlap)))
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride[0]) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride[1]) + 1)
    total_predictions = np.zeros((image_size[0],num_classes, image_size[2], image_size[3]))
    count_predictions = np.zeros((image_size[2], image_size[3]))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride[1]), int(row * stride[0])
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])
            img = image[:,:, y_min:y_max, x_min:x_max]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_prediction= model(normalize(padded_img),1)
            if isinstance(padded_prediction,tuple):
                padded_prediction=padded_prediction[0]
            if flip:
                fliped_predictions = model(padded_img.flip(-1))
                padded_prediction = 0.5 * (fliped_predictions.flip(-1) + padded_prediction)
            predictions = padded_prediction[:, :, :img.shape[2], :img.shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:,:, y_min:y_max, x_min:x_max] += predictions.data.cpu().numpy()

    total_predictions /= count_predictions
    return total_predictions


def multi_scale_predict(model, image, scales, num_classes, device, normalize,flip=False,dataloader=False):
    if dataloader:
        image=image*torch.tensor(normalize.std).to(device).view((1,3,1,1))+torch.tensor(normalize.mean).to(device).view((1,3,1,1))
        upsample = nn.Upsample(size=(image.shape[2],image.shape[3]), mode='bilinear', align_corners=True)
        total_predictions=torch.zeros((image.shape[0],num_classes,image.shape[2],image.shape[3]))
        for scale in scales:
            # scaled_img=F.interpolate(image,scale_factor=scale,mode='bicubic')
            scaled_img=torch.from_numpy(ndimage.zoom(image.cpu().numpy(),(1,1,scale,scale))).to(device)
            scaled_img=normalize(scaled_img)
            scaled_prediction=model(scaled_img)
            scaled_prediction=upsample(scaled_prediction)
            total_predictions+=scaled_prediction.cpu()
        total_predictions/=len(scales)
        return total_predictions



    else:
        image=np.asarray(image)
        input_size = (image.shape[0], image.shape[1])
        upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
        total_predictions = np.zeros((num_classes, image.shape[0], image.shape[1]))
        to_tensor=transforms.ToTensor()
        for scale in scales:
            scaled_img = ndimage.zoom(image, (float(scale), float(scale),1), order=1, prefilter=False)
            scaled_img = to_tensor(scaled_img).to(device)
            scaled_img=normalize(scaled_img).unsqueeze(0)
            scaled_prediction= upsample(model(scaled_img).cpu())

            if flip:
                fliped_img = scaled_img.flip(-1).to(device)
                fliped_predictions = upsample(model(fliped_img).cpu())
                scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
            total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

        total_predictions /= len(scales)
        return total_predictions

def main():
    args = parse_arguments()
    config = json.load(open(args.config))

    # Dataset used for training the model
    dataset_type = config['train_loader']['type']
    assert dataset_type in ['VOC', 'COCO', 'CityScapes', 'ADE20K','CityScapes_super_pixels']
    if dataset_type == 'CityScapes' or dataset_type =='CityScapes_super_pixels' :
        scales = [0.5,0.75, 1.0, 1.25, 1.5, 1.75]
    else:
        scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(loader.MEAN, loader.STD)
    num_classes = loader.dataset.num_classes
    palette = loader.dataset.palette

    # Model
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    device = torch.device('cuda:0' if args.device=='gpu' else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    # If during training, we used data parallel
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        # for gpu inference, use data parallel
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
        # for cpu inference, remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict
    # load
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    label_path = os.path.join('../data/cityscapes/', 'gtFine_trainvaltest', 'gtFine', 'val')
    image_path = args.images
    SUFIX='_gtFine_labelIds.png'
    assert os.listdir(image_path) == os.listdir(label_path)
    image_paths, label_paths = [], []
    for city in os.listdir(image_path):
        image_paths.extend(sorted(glob(os.path.join(image_path, city, '*.png'))))
        label_paths.extend(sorted(glob(os.path.join(label_path, city, f'*{SUFIX}'))))
    image_files=list(zip(image_paths,label_paths))[385:386]
    metric=metrics()
    with torch.no_grad():
        tbar = tqdm(image_files, ncols=100)
        j=0
        images = torch.zeros(args.bs, 3, 1024, 2048)
        labels = torch.zeros(args.bs,  1024, 2048)
        for img_file,label_file in tbar:
                image = Image.open(img_file).convert('RGB')
                l=np.asarray(Image.open(label_file), dtype=np.int32)
                image=to_tensor(image)
                label = to_tensor(np.asarray(Image.open(label_file), dtype=np.int32))
                for k, v in ID_TO_TRAINID.items():
                    label[label == k] = v
                    l[l==k]=v
                images[j,:,:,:]=image
                labels[j,:,:]=label
                if j!=args.bs-1:
                    j+=1
                    continue

            if args.mode == 'multiscale':
                prediction = multi_scale_predict(model, image, scales, num_classes, normalize=normalize,device=device)
            elif args.mode == 'sliding':
                prediction = sliding_predict(model, images, normalize,num_classes)
            seg_metrics = eval_metrics(
                torch.from_numpy(prediction).to(device),
                labels.to(device), 19)
            metric._update_seg_metrics(*seg_metrics)
        pixAcc, mIoU, classIoU, IoUs = metric._get_seg_metrics().values()
        print(mIoU)
        print(pixAcc)








class metrics():
    def __init__(self):
        self._reset_metrics()
    def _reset_metrics(self):
         self.batch_time = AverageMeter()
         self.data_time = AverageMeter()
         self.total_loss = AverageMeter()
         self.total_inter, self.total_union = 0, 0
         self.total_correct, self.total_label = 0, 0
         self.IoUs=[]
    def _update_seg_metrics( self,correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union
        self.IoUs.append( (1.0 * inter / (np.spacing(1) + union)).mean())
    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()

        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(19), np.round(IoU, 3))),
            "IoUs": self.IoUs
        }
def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='./saved/22_tests/mobilenet_deeplab_resnet50_balanced/1/config.json',type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='sliding', type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')
    parser.add_argument('-m', '--model', default='', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default='../data/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/', type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-e', '--extension', default='.png', type=str,
                        help='The extension of the images to be segmented')
    parser.add_argument('-de', '--device', default='cpu', type=str,
                        help='what device should be used for inference?')
    parser.add_argument('-d', '--device_id', default='', type=str,
                       help='indices of GPUs to enable (default: all)')
    parser.add_argument('-da', '--dataloader', default=False, type=bool,
                        help='using a dataloader?')
    parser.add_argument('-bs', '--bs', default=10, type=bool,
                        help='using a dataloader?')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    return args

if __name__ == '__main__':
    main()
