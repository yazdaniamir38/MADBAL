import numpy as np
import os
from glob import glob
import pickle
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import models
import cv2 as cv
ignore_label = 255


ID_TO_TRAINID = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                    3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                    7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                    14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                    18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                    28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}



class super_pixel_loader_vgg_padding():
    def __init__(self,root='../../data/cityscapes'):
        self.root_sp_info=os.path.join(root,'superpixels_quarter_16_16_raw/train/')
        self.root_img=os.path.join(root,'leftImg8bit_trainvaltest/leftImg8bit/train')
        self.files=glob(os.path.join(self.root_sp_info,"*.pkl"))
        self.idxes=list(range(len(self.files)))
        self.transform=T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    def __getitem__(self, item=None):
        self.idx_generator()
        file_name=self.files[self.idx]
        pickle_file = open(file_name, "rb")
        sp_dic = pickle.load(pickle_file)
        pickle_file.close()
        labels = sp_dic['labels']
        valid_idxes = sp_dic['valid_idxes']
        img_name=os.path.split(file_name)[1]
        city=img_name.split('_')[0]
        img_name=os.path.join(self.root_img,city,img_name[:-4])
        img=Image.open(img_name).convert("RGB").resize((labels.shape[-1::-1]))
        img=self.transform(img)
        self.shape=img.shape[1:]
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
        return out,sp_dic,file_name
    def map_clusters_to_img(self,clusters):
        file_name = self.files[self.idx]
        pickle_file = open(file_name, "rb")
        sp_dic = pickle.load(pickle_file)
        pickle_file.close()
        labels = sp_dic['labels']
        valid_idxes = sp_dic['valid_idxes']
        out=np.zeros(self.shape)
        for j in range(len(valid_idxes)):
            out[labels==valid_idxes[j]]=clusters[j]
        return out
    def idx_generator(self):
        assert len(self.idxes)
        self.idx=self.idxes.pop()

def extract_clusters():
    global vgg
    vgg= getattr(models, 'vgg16')('True').features[:-1]
    vgg.eval()
    sampler = super_pixel_loader_vgg_padding()
    pickle_file=open('../saved/finalized_model_cityscapes_vgg_padding_quarter_9clusters_16.sav','rb')
    model=pickle.load(pickle_file)
    pickle_file.close()

    for i in range(2975):
        result = sampler[None]
        samples, dic, name=result[0],result[1],result[2]
        print('processing:'+name)
        with torch.no_grad():
            samples=vgg(samples).cpu()
        samples=samples.squeeze().numpy()
        clustered_samples=model.predict(samples)
        # clustered_img=sampler.map_clusters_to_img(clustered_samples)
        # plt.imshow(clustered_img)
        # plt.show()
        dic['clusters']=clustered_samples
        pickle.dump(dic, open(name, 'wb'))


def assign_clusters_randomly(root='../AL/data/cityscapes/superpixels_quarter_16_16_measure'):
    files=glob(os.path.join(root,'train','*.pkl'))
    for file in files:
        dic=pickle.load(open(file,'rb'))
        len_clusters=len(dic['clusters'])
        dic['clusters']=np.random.randint(low=0,high=9,size=(len_clusters))
        pickle.dump(dic,open(file,'wb'))





if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    extract_clusters()
