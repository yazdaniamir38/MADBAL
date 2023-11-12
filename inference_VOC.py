import argparse
import scipy
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
import pickle
import matplotlib.pyplot as plt
import models
from utils.metrics import eval_metrics, AverageMeter
from utils.helpers import colorize_mask
from collections import OrderedDict
batch_time = AverageMeter()
data_time = AverageMeter()
total_loss = AverageMeter()
total_inter, total_union = 0, 0
total_correct, total_label = 0, 0
import cv2 as cv
# For cityscape only
ignore_label = 255
ID_TO_TRAINID = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                    3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                    7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                    14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                    18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                    28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
# def pad_image(img, added_size):
#     # rows_to_pad = max(target_size[0] - img.shape[1], 0)
#     # cols_to_pad = max(target_size[1] - img.shape[2], 0)
#     padded_img = F.pad(img, (added_size[1][0], added_size[1][1], added_size[0][0], added_size[0][1]), "reflect")
#     return padded_img

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[1], 0)
    cols_to_pad = max(target_size[1] - img.shape[2], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img
def sliding_predict(model, image, normalize,num_classes,flip=False):
    to_tensor=transforms.ToTensor()
    # image=np.asarray(image,dtype=np.float32)
    # image_size=image.shape
    # resize_factor = image_size[0]/768  if image_size[0] > image_size[1] else image_size[1] / 768
    # h, w = int(image_size[0] // resize_factor), int(image_size[1] // resize_factor)
    # image = cv.resize(image, (w, h), interpolation=cv.INTER_LINEAR)
    # h_1,w_1=(768-(h%768)),(768-(w%768))
    # to_add=[((h_1+1)//2,h_1//2),((w_1+1)//2,w_1//2)]
    image=to_tensor(image)
    # image = pad_image(image, to_add)
    image_size_1 = image.shape
    # h=image_size[1]+256-(image_size[1]%256)
    # w = image_size[2] + 256 - (image_size[2] % 256)
    # image=pad_image(image,[h,w])

    tile_size = (image_size_1[1], image_size_1[2])
    overlap = 0

    stride = ceil(tile_size[0] * (1 - overlap))
    
    num_rows = int(ceil((image_size_1[1] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size_1[2] - tile_size[1]) / stride) + 1)
    total_predictions = torch.zeros((1,num_classes, image_size_1[1], image_size_1[2]))
    count_predictions = np.zeros((image_size_1[1], image_size_1[2]))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size_1[2])
            y_max = min(y_min + tile_size[0], image_size_1[1])

            img = image[:, y_min:y_max, x_min:x_max]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_prediction= model(normalize(padded_img).unsqueeze(0),2)
            if isinstance(padded_prediction, tuple):
                padded_prediction = padded_prediction[0]
            if flip:
                # fliped_img = padded_img.flip(-1)
                fliped_predictions = model(padded_img.flip(-1))
                padded_prediction = 0.5 * (fliped_predictions.flip(-1) + padded_prediction)
            predictions = padded_prediction[:, :, :img.shape[1], :img.shape[2]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:,:, y_min:y_max, x_min:x_max] += predictions.cpu()

    total_predictions /= count_predictions
    # total_predictions=total_predictions[:,:,to_add[0][0]:-to_add[0][1],to_add[1][0]:-to_add[1][1]]
    # total_predictions=F.interpolate(total_predictions,size=image_size[:-1],mode='nearest')
    return total_predictions.data.numpy().squeeze(0)


def multi_scale_predict(model, image, scales, num_classes, device, flip=False):
    input_size = (image.size(2), image.size(3))
    upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

    image = image.data.data.cpu().numpy()
    for scale in scales:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
        scaled_img = torch.from_numpy(scaled_img).to(device)
        scaled_prediction = upsample(model(scaled_img).cpu())

        if flip:
            fliped_img = scaled_img.flip(-1).to(device)
            fliped_predictions = upsample(model(fliped_img).cpu())
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions


def save_images(image, mask,label, output_path, image_file, palette):
	# Saves the image, the model output and the results after the post processing
    w, h = image.size
    # image_file = os.path.basename(image_file).split('.')[0]
    mask=np.argmax(mask.squeeze(),0)
    mask[label==255]=255
    colorized_mask = colorize_mask(mask, palette)
    colorized_mask.save(os.path.join(output_path, image_file+'.png'))

    colorized_gt=colorize_mask(label,palette)
    colorized_gt.save(os.path.join(output_path,image_file+'_gt.png'))
    # output_im = Image.new('RGB', (w*2, h))
    # output_im.paste(image, (0,0))
    # output_im.paste(colorized_mask, (w,0))
    # output_im.save(os.path.join(output_path, image_file+'_colorized.png'))
    # mask_img = Image.fromarray(mask, 'L')
    # mask_img.save(os.path.join(output_path, image_file+'.png'))

def main():
    args = parse_arguments()
    config = json.load(open(args.config))
    if args.device=='gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    # Dataset used for training the model
    dataset_type = config['train_loader']['type']
    assert dataset_type in ['VOC', 'COCO', 'CityScapes', 'ADE20K','CityScapes_super_pixels','VOC_superpixels']
    if dataset_type == 'CityScapes' or dataset_type =='CityScapes_super_pixels' :
        scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] 
    else:
        scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(loader.MEAN, loader.STD)
    num_classes = loader.dataset.num_classes
    palette = loader.dataset.palette

    # Model
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
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

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # image_files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))
    img_dir_name = '../data/VOCdevkit/VOC2012/'
    file_list = os.path.join(img_dir_name, "ImageSets/Segmentation", 'val' + ".txt")
    files= [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))][1008:1009]
    image_path=img_dir_name+'JPEGImages'
    label_path = img_dir_name+'SegmentationClass'
    metric=metrics()
    with torch.no_grad():
        tbar = tqdm(files, ncols=100)
        for i,file_name in  enumerate(tbar):
            image = Image.open(os.path.join(image_path,file_name[0]+'.jpg')).convert('RGB')
            label = to_tensor(np.asarray(Image.open(os.path.join(label_path,file_name[0]+'.png')), dtype=np.int32))
            # for k, v in ID_TO_TRAINID.items():
            #     label[label == k] = v
            # input = normalize(to_tensor(image)).unsqueeze(0)
            
            if args.mode == 'multiscale':
                prediction = multi_scale_predict(model, image, scales, num_classes, device)
            elif args.mode == 'sliding':
                prediction = sliding_predict(model, image, normalize,num_classes)
            else:
                prediction,_,_,_,_,_ = model(input.to(device))
            seg_metrics = eval_metrics(torch.from_numpy(prediction).unsqueeze(0).to(device), label.to(device), prediction.shape[0])
            metric._update_seg_metrics(*seg_metrics)
            print(list(metric._get_seg_metrics().values())[1])
            # prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
            save_images(image, prediction,label.squeeze().cpu().numpy(), args.output, file_name[0], palette)
        pixAcc, mIoU, classIoU, IoUs = metric._get_seg_metrics().values()
        print(mIoU)
        print(pixAcc)
        # pickle.dump(IoUs, open(os.path.join(args.output, 'IoUs.pkl'), 'wb'))
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
    parser.add_argument('-c', '--config', default='saved/resnet50/1/config.json',type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='sliding', type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')
    parser.add_argument('-m', '--model', default='saved/resnet50/1/checkpoint-epoch124.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default='data/VOC/jpegImages/', type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='resnet50_10', type=str,
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='.png', type=str,
                        help='The extension of the images to be segmented')
    parser.add_argument('-de', '--device', default='gpu', type=str,
                        help='what device should be used for inference?')
    parser.add_argument('-d', '--device_id', default='1', type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    main()
