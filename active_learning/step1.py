# step1 of active learning:
# evaluate the most recent trained model on all of the samples
# in unlabelled pool and extract their raw score, dominant class and class distribution
import torch
import numpy as np
import os
import pickle
import gc
from tqdm import tqdm
import torch.nn.functional as F
from torch.distributions import Categorical
from base import DataPrefetcher
from utils.metrics import JSD
#initial evaluation
def evaluate(model,dataloader,device,n_clusters,num_processes=2):
    model.eval()
    class_dist=np.zeros((dataloader.dataset.num_classes,n_clusters))
    cluster_dist=[[set() for i in range(n_clusters)],np.zeros((n_clusters)),np.zeros((n_clusters))]
    unlabelled_dir=os.path.split(dataloader.dataset.files[0])[0]
    dataloader=DataPrefetcher(dataloader,device=device,unlabelled=True)
    dataloader=tqdm(dataloader,ncols=130)
    with torch.no_grad():
        for  (count,(data,edges_mask,names,idx)) in enumerate(dataloader):

            logits,middle1,middle2,middle3,W,lossy_area=model(data)
            logits=F.interpolate(logits,scale_factor=0.25,mode='bilinear', align_corners=False)
            middle1=F.interpolate(middle1,scale_factor=0.25,mode='bilinear', align_corners=False)
            middle2=F.interpolate(middle2, scale_factor=0.25, mode='bilinear', align_corners=False)
            middle3=F.interpolate(middle3,scale_factor=0.25,mode='bilinear', align_corners=False)
            W=F.interpolate(W,scale_factor=0.25,mode='bilinear', align_corners=False)
            lossy_area=F.interpolate(lossy_area,scale_factor=0.25,mode='bilinear', align_corners=False)
            edges_mask=(F.interpolate(edges_mask.unsqueeze(1),scale_factor=0.25,mode='nearest')).squeeze()
            logits=F.softmax(logits,dim=1)
            middle1=F.softmax(middle1,dim=1)
            middle2 = F.softmax(middle2, dim=1)
            middle3 = F.softmax(middle3, dim=1)
            class_dist, cluster_dist = initial_extract(logits, middle1,
                                                           middle2,middle3, edges_mask,
                                                           lossy_area, W,names,
                                                           class_dist, cluster_dist, unlabelled_dir)

            del logits,lossy_area,W,middle1,middle2,middle3
            torch.cuda.empty_cache()
            gc.collect()
    class_dist=class_dist / class_dist.sum(axis=0,keepdims=True)
    if not os.path.exists(os.path.join(unlabelled_dir,'cls_dist')):
        os.makedirs(os.path.join(unlabelled_dir,'cls_dist'))
    pickle.dump(class_dist, open(os.path.join(unlabelled_dir,'cls_dist','class_dist.pkl'), 'wb'))
    pickle.dump(cluster_dist,open(os.path.join(unlabelled_dir,'cls_dist','cluster_dist.pkl'),'wb'))
    return class_dist,cluster_dist

def initial_extract(logits,middle1,middle2,middle3,edges_mask,lossies,W,names,class_dist,cluster_dist,unlabelled_dir):
    JS = JSD()
    for i,(name,logit) in enumerate(zip(names,logits)):
        logits_enthropy = Categorical(probs=logit.permute([1, 2, 0])).entropy()
        lossy = torch.exp(lossies[i])
        score_total=logits_enthropy+(W[i,0,:,:]*JS(logit, middle1[i]).sum(dim=0) +W[i,1,:,:]*JS(logit, middle2[i]).sum(dim=0)+W[i,2,:,:]*JS(logit, middle3[i]).sum(dim=0))
        score_total =  (-1 * edges_mask[i] + 1) * score_total * lossy[1]+edges_mask[i]*score_total*lossy[0]
        file_name=os.path.join(unlabelled_dir,name+'.pkl')
        pickle_file=open(file_name,'rb')
        dic=pickle.load(pickle_file)
        pickle_file.close()
        dic.pop('raw_score',None)
        dic.pop('dom_cls',None)
        labels=dic['labels']
        valid_idxes=dic['valid_idxes']
        clusters=dic['clusters']
        _,clses=torch.sort(logit,dim=0)
        clses=clses[-2:,:,:]
        dic['raw_score']=np.zeros((len(valid_idxes)))
        dic['dom_cls']=np.zeros((len(valid_idxes)))
        dic['highest']=np.zeros((len(valid_idxes)))
        for j,region_idx in enumerate(valid_idxes):
            mask=labels==region_idx
            cls=clses[-1,mask]
            raw_score=score_total[mask]
            dic['highest'][j] =torch.argmax(raw_score).item()
            raw_score=raw_score.mean()
            dom_cls,counts=torch.unique(cls,return_counts=True)
            cluster_dist[0][clusters[j]].update(dom_cls.tolist())
            cluster_dist[1][clusters[j]]+=len(cls)
            cluster_dist[2][clusters[j]]+=raw_score.item()
            dom_cls=dom_cls[torch.argmax(counts)]
            dic['raw_score'][j]=raw_score.item()
            dic['dom_cls'][j]=dom_cls.item()
            class_dist[dom_cls,clusters[j]]+=1
        pickle.dump(dic, open(file_name, 'wb'))
    del logits_enthropy,score_total,edges_mask
    torch.cuda.empty_cache()
    return class_dist,cluster_dist
