import numpy as np
import os
from glob import glob
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models
import cv2 as cv





class super_pixel_loader_vgg_padding():
    def __init__(self,root='../../data/cityscapes'):
        self.root_sp_info = os.path.join(root, 'superpixels_quarter_16_16/train')
        self.root_img = os.path.join(root, 'leftImg8bit_trainvaltest/leftImg8bit/train')
        self.files=glob(os.path.join(self.root_sp_info,"*.pkl"))
        self.idx=list(range(len(self.files)))
        self.transform=T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    def __getitem__(self, item=None):
        idx=self.idx_generator()
        # idx=2851
        file_name=self.files[idx]
        # print(idx)
        pickle_file = open(file_name, "rb")
        sp_dic = pickle.load(pickle_file)
        pickle_file.close()
        labels = sp_dic['labels']
        valid_idxes = sp_dic['valid_idxes']
        img_name=os.path.split(file_name)[1]
        city=img_name.split('_')[0]
        # img_name=os.path.join(self.root_img,img_name[:-4]+'.jpg')
        img_name = os.path.join(self.root_img,city, img_name[:-4] )
        # img=plt.imread(img_name)
        img=Image.open(img_name).convert("RGB").resize((labels.shape[-1::-1]))
        img=self.transform(img)

        out = torch.empty((len(valid_idxes),3, 16,16))
        j = 0
        for i in range(0, len(valid_idxes)):
            y, x = np.where(labels == valid_idxes[i])
            rangey,rangex=np.ptp(y),np.ptp(x)
            ymin,xmin=y.min(),x.min()
            patch=np.ones((rangey+1,rangex+1,3))*img[:,y,x].mean(axis=(1)).numpy()
            patch[y-ymin,x-xmin,:] = img[:,y, x].permute([1,0]).numpy()
            if patch.shape[0]!=16 or patch.shape[1]!=16:
                patch=cv.resize(patch, dsize=(16, 16), interpolation=cv.INTER_CUBIC)
            patch=torch.from_numpy(patch).permute([2,0,1])
            assert patch.shape[1] == 16 and patch.shape[2] == 16
            out[j, :,:,:] = patch
            j += 1
        return out
    def idx_generator(self):
        assert len(self.idx)
        idx=self.idx.pop()
        return idx




os.environ["CUDA_VISIBLE_DEVICES"] = '0'
global vgg
vgg= getattr(models, 'vgg16')('True').features[:-1].cuda()
vgg.eval()

rng = np.random.RandomState(0)
kmeans = MiniBatchKMeans(n_clusters=9,random_state=rng,batch_size=2*512,max_iter=15,verbose=True)
pca = PCA(1024)
for k in range(6):
    sampler = super_pixel_loader_vgg_padding()
    for m in range(2975 // 2):
        for j in range(2):
            if j==0:
                weights=sampler[None]
            else:
                weights=torch.cat([weights,sampler[None]],dim=0)
        with torch.no_grad():
            weights=vgg(weights.cuda()).cpu()
        weights = weights.squeeze().detach().numpy()
        kmeans.partial_fit(weights)
        if m%5==0:
            print("Partial fit of %4i out of %i" % (m, 1464))
    print("iteration %i is done." %(k+1))
filename = '../saved/finalized_model_cityscapes_vgg_padding_quarter_9clusters_16.sav'
pickle.dump(kmeans, open(filename, 'wb'))

