import os
import cv2 as cv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import exposure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='Cityscapes')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--sp_method', type=str, default='seeds')
parser.add_argument('--num_superpixels', type=int, default=2048)
parser.add_argument('--resize_factor', type=int, default=1)

FLAGS = parser.parse_args()


def extract_superpixel_slic(image_name_path):
    method = 'slic'

    num_superpixels = FLAGS.num_superpixels
    sigma = 0
    superpixel_label_dir = './superpixels/{}/{}_{}/{}/label'.format(FLAGS.dataset_name, method, num_superpixels,
                                                                    FLAGS.split)
    if not os.path.exists(superpixel_label_dir):
        os.makedirs(superpixel_label_dir)
    superpixel_result_dir = './superpixels/{}/{}_{}/{}/result'.format(FLAGS.dataset_name, method, num_superpixels,
                                                                      FLAGS.split)
    if not os.path.exists(superpixel_result_dir):
        os.makedirs(superpixel_result_dir)

    max_n = 0
    nr_sample = 0
    for image_name, image_path in image_name_path.items():

        print(image_name)
        img = plt.imread(image_path)
        img_eq = exposure.equalize_hist(img)

        labels = slic(img_eq, n_segments=num_superpixels, sigma=sigma)
        result = mark_boundaries(img, labels)

        output_dic = {}
        output_dic['labels'] = labels.astype(np.int16)
        num_sp = labels.max() + 1

        if num_sp > max_n:
            max_n = num_sp

        output_dic['valid_idxes'] = np.unique(labels)

        pickle.dump(output_dic, open(os.path.join(superpixel_label_dir, image_name + '.pkl'), 'wb'))
        plt.imsave(os.path.join(superpixel_result_dir, image_name + '.jpg'), result)
        nr_sample += 1
    print(max_n)

def _compute_base_size(size_base, h, w):
    if w > h:
        h = int(float(h) / w * size_base)
        w = size_base
    else:
        w = int(float(w) / h * size_base)
        h = size_base
    return h, w


def extract_superpixel_seeds(image_name_path):

    prior = 3
    num_levels = 5
    num_histogram_bins = 10
    num_superpixels = FLAGS.num_superpixels
    if FLAGS.dataset_name=='cityscapes':
        superpixel_label_dir = '../../data/cityscapes/superpixels_quarter_16_16_raw/{}/'.format(FLAGS.split)
        if not os.path.exists(superpixel_label_dir):
            os.makedirs(superpixel_label_dir)
        superpixel_result_dir = '../../data/cityscapes/superpixels_quarter_16_16_raw/{}/result'.format(FLAGS.split)
    else:
        superpixel_label_dir = '../../data/VOCdevkit/superpixels_400_measure_raw/{}/'.format(FLAGS.split)
        if not os.path.exists(superpixel_label_dir):
            os.makedirs(superpixel_label_dir)
        superpixel_result_dir = '../../data/VOCdevkit/superpixels_400_measure_raw/{}/result'.format(FLAGS.split)
    if not os.path.exists(superpixel_result_dir):
        os.makedirs(superpixel_result_dir)

    max_n = 0
    nr_sample = 0
    for image_name, image_path in image_name_path.items():

        print(image_name)
        img = Image.open(image_path)
        width, height = img.size

        # num_superpixels=(width*height)//1024
        if FLAGS.dataset_name=='cityscapes':
            resize_factor=4
        else:
            resize_factor = width/400 if width>height else height/400
        img = img.convert('RGB').resize((int(width // resize_factor), int(height // resize_factor)))
        width, height = img.size

        num_superpixels = (width * height) // 256
        img_eq = exposure.equalize_hist(np.asarray(img))

        converted_img = cv.cvtColor(img_eq.astype(np.float32), cv.COLOR_RGB2HSV)
        height, width, channels = converted_img.shape
        seeds = cv.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, prior,
                                                  num_histogram_bins, True)
        seeds.iterate(converted_img, 10)

        labels = seeds.getLabels()

        output_dic = {}
        output_dic['labels'] = labels.astype(np.int16)
        num_sp = labels.max() + 1

        if num_sp > max_n:
            max_n = num_sp

        output_dic['valid_idxes'] = np.unique(labels)

        pickle.dump(output_dic, open(os.path.join(superpixel_label_dir, image_name + '.pkl'), 'wb'))
        nr_sample += 1
    print(max_n)



if __name__ == '__main__':

    if FLAGS.dataset_name == 'pascal_voc_seg':
        devkit_path = '../../data/VOCdevkit/'
        image_dir = devkit_path + 'VOC2012/JPEGImages'
        imageset_path = devkit_path + 'VOC2012/ImageSets/Segmentation/%s.txt' % FLAGS.split
        with open(imageset_path, 'r') as f:
            lines = f.readlines()
        image_list = [x.strip() for x in lines]
    elif FLAGS.dataset_name == 'cityscapes':
        devkit_path = '../../data/cityscapes/'
        image_dir = devkit_path + 'leftImg8bit_trainvaltest/leftImg8bit/train'
        image_paths, label_paths = [], []
        for city in os.listdir(image_dir):
            image_paths.extend(sorted(glob(os.path.join(image_dir, city, '*.png'))))
    image_name_path = {}

    if FLAGS.dataset_name == 'pascal_voc_seg':
        for image_name in image_list:
            image_path = os.path.join(image_dir, image_name + '.jpg')
            image_name_path[image_name] = image_path
    elif FLAGS.dataset_name == 'cityscapes':
        for image_path in image_paths:
            image_name_path[os.path.split(image_path)[1]] = image_path
    elif FLAGS.sp_method == 'seeds':
        extract_superpixel_seeds(image_name_path)
    elif FLAGS.sp_method == 'slic':
        extract_superpixel_slic(image_name_path)
    else:
        print('%s not implemented' % FLAGS.sp_method)
        raise RuntimeError
