#The second step of active learning
# we extract from each cluster samples with highest uncertatinty score and update data loaders

import numpy as np
import pickle
import os
import heapq
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import random
import cv2 as cv


TrainID_to_ID = {0: 7, 1: 8, 2: 11, 3: 12, 4: 13,
                    5: 17, 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24, 12: 25, 13: 26, 14: 27,
                    15: 28, 16: 31, 17: 32, 18: 33}

def extract_samples(class_dist,k,portion,unlabelled_path='../data/cityscapes/superpixels/unlabelled_pool'):
    files=glob(os.path.join(unlabelled_path,'*.pkl'))[portion[0]:portion[1]]

    samples=dict()
    for file_number,file in enumerate(tqdm(files,ncols=130)):
        pickle_file=open(file,'rb')
        dic=pickle.load(pickle_file)
        pickle_file.close()
        name=os.path.split(file)[1][:-4]
        samples[name]=[MyHeap() for i in range(len(class_dist))]
        raw_scores=dic['raw_score']
        dom_cls=dic['dom_cls'].astype('uint8')
        clusters=dic['clusters']
        dist = np.histogram(clusters, bins=np.arange(len(k)+1))[0]
        avails = distribute(dist, k)
        for i,raw_score in enumerate(raw_scores):
            score=raw_score*np.exp(-1*class_dist[int(dom_cls[i]),clusters[i]])

            samples[name][clusters[i]].push([score,dic['valid_idxes'][i],dom_cls[i]])
            if len(samples[name][clusters[i]]._data)>avails[clusters[i]]:
                samples[name][clusters[i]].pop()
        l=[]
        ([l.extend([samples[name][i]._data[j][1:] for j in range(avails[i])]) for i in range(9)])
        samples[name]=np.asarray(l)
    return samples

#name format if different should be adjusted accordingly
def update_loaders(names,unlabelled_names,regions):

    for name in regions.keys():
        pickle_file=open(os.path.join(os.path.split(unlabelled_names[0])[0],name+'.pkl'),'rb')
        dic=pickle.load(pickle_file)
        pickle_file.close()
        valid_idxes=dic['valid_idxes']
        lbls=dic['labels']
        highest=dic['highest']
        gt=np.asarray(Image.open(os.path.join('../data/cityscapes/gtFine_trainvaltest/gtFine/train',os.path.split(name)[1].split('_')[0],name[:-4].split('leftImg8bit')[0]+'gtFine'+'_labelIds.png')),dtype=np.int32)
        gt = cv.resize(gt, lbls.shape[-1::-1], interpolation=cv.INTER_NEAREST)
        if os.path.join(os.path.split(names[0])[0],name[:-4].split('leftImg8bit')[0]+'gtFine'+'_labelIds.png') in names:
            label=np.asarray(Image.open((os.path.join(os.path.split(names[0])[0],name[:-4].split('leftImg8bit')[0]+'gtFine'+'_labelIds.png'))),dtype=np.int32)
        else:
            label=np.ones_like(lbls)*255.0
            names.append(os.path.join(os.path.split(names[0])[0],name[:-4].split('leftImg8bit')[0]+'gtFine'+'_labelIds.png'))
        indices_to_remove = []

        for (region_index,cls) in regions[name]:
            mask=lbls==region_index
            index = np.where(valid_idxes == region_index)[0]
            highest_index=np.where(mask)
            if len(highest_index[0])==1:
                indices_to_remove.append(index)
            highest_index=(highest_index[0][int(highest[index])],highest_index[1][int(highest[index])])
            label[highest_index]=gt[highest_index]
            lbls[highest_index]=-1

        if not len(valid_idxes):
            os.remove(os.path.join(os.path.split(unlabelled_names[0])[0],name,'.pkl'))
            unlabelled_names.remove(os.path.join(os.path.split(unlabelled_names[0])[0],name,'.pkl'))
        if len(indices_to_remove):
            dic['valid_idxes']=np.delete(valid_idxes,indices_to_remove)
            dic['clusters']=np.delete(dic['clusters'],indices_to_remove)
        else:
            dic['valid_idxes']=valid_idxes
            dic['labels']=lbls
        pickle.dump(dic,open(os.path.join(os.path.split(unlabelled_names[0])[0], name+'.pkl'),'wb'))
        label=Image.fromarray(label.astype('uint8'))
        label.save(os.path.join(os.path.split(names[0])[0],name[:-4].split('leftImg8bit')[0]+'gtFine'+'_labelIds.png'))
    return names,unlabelled_names

def distribute(dist,num):
    avail=np.zeros_like(dist)+num
    count=0
    flag=0
    while (avail>dist).sum()>0:
        if count<150:
            mask=avail>dist
            value=((avail[mask]-dist[mask]).sum())/((~mask).sum())
            avail[~mask]+=int(value)
            index = np.random.randint(0, (~mask).sum(), size=1)
            index = np.where(~mask)[0][index]
            avail[index] += (int(value * (~mask).sum()) - int(value) * (~mask).sum())
            avail[mask]=dist[mask]
            count+=1
        else:
            flag=1
            break

    if flag:
         print('Assigning the distribution is not possible.')
         raise SystemExit
    else:
        return avail

class MyHeap(object):
   def __init__(self, initial=None, key=lambda x:x[0]):
       self.key = key
       self.index = 0
       if initial:
           self._data = [(item) for i, item in enumerate(initial)]
           self.index = len(self._data)
           heapq.heapify(self._data)
       else:
           self._data = []

   def push(self, item):
       heapq.heappush(self._data, (item))
       self.index += 1

   def pop(self):
       return heapq.heappop(self._data)




