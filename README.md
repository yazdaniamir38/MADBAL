# MADBAL [[BMVC 2023 Oral](https://bmvc2023.org)]
This is an official implementation of the paper "Maturity-Aware Active Learning for Semantic Segmentation with Hierarchically-Adaptive Sample Assessment."


[[Project page]](http://signal.ee.psu.edu/research/MADBAL.html)
[[Paper]](https://arxiv.org/abs/2104.06394 "Paper")


### Table of contents
* [Installation](#installation)
* [Benchmark results](#benchmark-results)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

### Installation
##### Prerequisites
Our code is based on `Python 3.9.12` and uses the following Python packages.
```
pytorch=1.11.0
numpy=1.21.2
matplotlib=3.5.1
opencv=4.5.5
scikit-learn=1.0.2
scikit-image=0.19.2
tqdm=4.63.1
scipy=1.8.0
pillow=9.0.1
imageio=2.9.0
torchvision=0.12.0
tensorboard=2.8.0
```


##### Clone this repository
```shell
git clone https://github.com/yazdaniamir38/MADBAL.git
cd MADBAL
```

##### Download dataset
We conducted our experiments on the following datasets:

* For __Cityscapes__, first visit the link and login to download. Once downloaded, you need to unzip it. It's worth mentioning that we trained and validated on quarter resolution samples. [[Cityscapes]](https://www.cityscapes-dataset.com "cityscapes")

* For __PASCAL VOC 2012__, the dataset will be automatically downloaded via `torchvision.datasets.VOCSegmentation`. 
##### AL_preprocess
includes the scripts to preprocess the data for AL.

__“extract_superpixels.py”__: we extract superpixels via SEEDS algorithm and store the details in dictionaries with different keys like: “valid indices” and “labels”.

__“clustering.py”__: we fit a K-means clustering model on the superpixels. The superpixels are first fitted into a rectangular patch with size 16*16 (look at the class “super_pixel_loader_vgg_padding” for more details) and then fed to a pretrained VGG16 feature extraction network. The output is a vector of 512*1 which will be used by K-means.

__“assign_clusters.py”__: once the clustering model is trained, the superpixels are assigned to a cluster. This information is added to their corresponding dictionary with the key “cluster”.
##### Train
__“train.py”__: the main script to be called when initiating MADBAL, based on given parameters in the “config.json” file, it initiates the process and handles the switching between different stages such as training phase I, training Phase II and active sample assessment.

__“trainer.py”__: the actual forward propagation and backward propagation for both phase I and II as well as validation happen here.

__“models”__: includes the scripts with class defining the architecture of our model with different backbones. 
##### Active learning
includes the scrips for active sample assessment step.

__“step1.py”__: we feed all the samples to the trained model and store the uncertainty
scores of the pixels, superpixels and clusters.

__“step2.py”__: based on the calculated scores and assigned budgets to clusters, for each image we select superpixels with highest uncertainty scores, and within the selected superpixels, we select the most uncertain pixels, label them, and add them to the pool.
##### Inference
__“inference_cityscapes.py”/“inference_VOC.py”__: The scripts to test a trained model on a dataset with different methods such as sliding or multiscale predict (see the script for more details).

Please download our trained model weights from [here](https://drive.google.com/drive/folders/1L-g6uoNK5kM7LAEvDX6J6NiBsNcBCEfT?usp=share_link). Once downloaded, store the weights in "checkpoint/" and run:
```shell
python inference_VOC.py --model checkpoints/weights_name.pth
python inference_cityscapes.py --model checkpoints/weights_name.pth
```


### Benchmark results
We report the average ± one std of mean IoU of 3 runs for both datasets.
##### Cityscapes
model|backbone (encoder)| # labelled pixels per img (% annotation) | mean IoU (%)
:---|:---|:---:|:---:
MADBAL|MobileNetv2|20 (0.015)|47.5 ± 0.5
MADBAL|MobileNetv2|40 (0.031)|59.0 ± 0.3
MADBAL|MobileNetv2|60 (0.046)|61.5 ± 0.4
MADBAL|MobileNetv2|80 (0.061)|62.7 ± 0.2
MADBAL|MobileNetv2|100 (0.076)|63.6 ± 0.2
Fully-supervised|MobileNetv2|256x512 (100)| 66.5 ± 0.6
MADBAL|MobileNetv3|20 (0.015)|49.1 ± 0.4
MADBAL|MobileNetv3|40 (0.031)|57.6 ± 0.2
MADBAL|MobileNetv3|60 (0.046)|59.3 ± 0.3
MADBAL|MobileNetv3|80 (0.061)|62.3 ± 0.2
MADBAL|MobileNetv3|100 (0.076)|62.8 ± 0.1
Fully-supervised|MobileNetv2|256x512 (100)| 68.5 ± 0.4
MADBAL|ResNet50|20 (0.015)|51.5 ± 0.5
MADBAL|ResNet50|40 (0.031)|63.3 ± 0.2
MADBAL|ResNet50|60 (0.046)|66.7 ± 0.3
MADBAL|ResNet50|80 (0.061)|67.2 ± 0.1
MADBAL|ResNet50|100 (0.076)|68.4 ± 0.3
Fully-supervised|ResNet50|256x512 (100)|72.0 ± 0.3

##### PASCAL VOC 2012
model|backbone (encoder)| # labelled pixels per img (% annotation) | mean IoU (%)
:---|:---|:---:|:---:
MADBAL|MobileNetv3|10 (0.009)|36.0 ± 0.6
MADBAL|MobileNetv3|20 (0.017)|60.3 ± 0.5
MADBAL|MobileNetv3|30 (0.026)|63.0 ± 0.4
MADBAL|MobileNetv3|40 (0.034)|63.6 ± 0.3
Fully-supervised|MobileNetv3|N/A (100)|65.1 ± 0.5
MADBAL|ResNet50|10 (0.009)|67.0 ± 0.7
MADBAL|ResNet50|20 (0.017)|72.4 ± 0.4
MADBAL|ResNet50|30 (0.026)|73.3 ± 0.5
MADBAL|ResNet50|40 (0.034)|74.3 ± 0.1
Fully-supervised|ResNet50|N/A (100)|76.1 ± 0.4
### Citation
```shell
@inproceedings{Yazdani_2023_BMVC,
author    = {Yazdani, Amirsaeed and Li, Xuelu and Monga, Vishal},
title     = {Maturity-Aware Active Learning for Semantic Segmentation with Hierarchically-Adaptive Sample Assessment},
booktitle = {34th British Machine Vision Conference 2022, {BMVC} 2023, Aberdeen, UK, November 20-24, 2023},
publisher = {{BMVA} Press},
year      = {2023}
}
```

### Acknowledgements
We borrowed codes heavily from https://github.com/yassouali/pytorch-segmentation and partially from https://github.com/NoelShin/PixelPick and https://github.com/cailile/Revisiting-Superpixels-for-Active-Learning.

If you need further details feel free to reach me at yazdaniamirsaeed@gmail.com.
