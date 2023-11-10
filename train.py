import os
import json
import argparse
import torch
import dataloaders
from active_learning import evaluate,initial_extract,extract_samples,update_loaders
import models
import numpy as np


from utils import losses
from utils import Logger
from trainer import Trainer
import pickle
import gc

def get_instance(module, name, config,*args,idx_remove=False):
    if idx_remove:
        return getattr(module, config['type'])(*args,**config['args'])
    else:
        return getattr(module, config[name]['type'])(*args,**config[name]['args'])


def main(config, resume,cont,sample_selection,cls_dist):
    train_logger = Logger()
    stage=args.stage
    # DATA LOADERS
    train_loader = get_instance(dataloaders, 'train_loader', config)
    train_loader2 = get_instance(dataloaders, 'train_loader2', config)
    train_loader3=get_instance(dataloaders, 'train_loader3', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)
    val_loader_2=get_instance(dataloaders, 'val_loader_2', config)
    unlabelled_loader = get_instance(dataloaders, 'active_learner', config)

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])
    loss_no_mean=getattr(losses, config['loss'])(ignore_index = config['ignore_index'],reduction='none')
    step=0
    while step<5:
        model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
        step += 1
        # TRAINING
        trainer = Trainer(
            model=model,
            loss=loss,
            step=step,
            loss2=loss_no_mean,
            resume=resume,
            config=config,
            train_loader=train_loader,
            train_loader2=train_loader2,
            train_loader3=train_loader3,
            val_loader=val_loader,
            val_loader_2=val_loader_2,
            train_logger=train_logger,
        thresh=0,
        stage=stage)
        # train the network only
        if not sample_selection:
            if stage==1:
                trainer.train_1()
            if os.path.exists(os.path.join(trainer.checkpoint_dir, 'best_model.pth')):
                trainer._resume_checkpoint(os.path.join(trainer.checkpoint_dir,'best_model.pth'),init_lossy=1)
            if step<5:
                trainer.get_thresh()
                epoch=trainer.start_epoch
                trainer.epochs=epoch+config['trainer']['epochs2']-1
                trainer.train_2()
                torch.cuda.empty_cache()
                gc.collect()
        stage=1
        if os.path.exists(os.path.join(trainer.checkpoint_dir, 'best_model.pth')):
            trainer._resume_checkpoint(os.path.join(trainer.checkpoint_dir, 'best_model.pth'))

        sample_selection=0
        torch.backends.cudnn.benchmark = True
        if step!=5:
            if len(cls_dist):
                pickle_file=open(os.path.join(cls_dist,'class_dist.pkl'),'rb')
                class_dist=pickle.load(pickle_file)
                pickle_file.close()
                pickle_file = open(os.path.join(cls_dist,'cluster_dist.pkl'), 'rb')
                cluster_dist = pickle.load(pickle_file)
                pickle_file.close()
                cls_dist=''
            else:
                class_dist,cluster_dist=evaluate(trainer.model,unlabelled_loader,trainer.device,config['n_clusters'])
            cluster_dist=cluster_dist[2]/cluster_dist[1]
            cluster_dist=cluster_dist*config['n_samples']//cluster_dist.sum()
            if config['n_samples']-cluster_dist.sum()>0:
                cluster_dist[np.argmin(cluster_dist)]+=(config['n_samples']-cluster_dist.sum())
            cluster_dist=cluster_dist.astype('int')
            regions = extract_samples(class_dist, cluster_dist,
                                                               [0, len(unlabelled_loader.dataset.files)],
                                                               os.path.split(
                                                                   unlabelled_loader.dataset.files[0])[0])
            train_loader.dataset.files,unlabelled_loader.dataset.files=update_loaders(train_loader.dataset.files,unlabelled_loader.dataset.files,regions)

            train_loader2.dataset.files=train_loader.dataset.files
            print('selection_done! step:{},train_size:{}'.format(step, len(trainer.train_loader.dataset.files)))



if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config_resnet50.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default='', type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default='0,1', type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('--stage', default=2, type=int,
                        help='which stage to start with?')
    parser.add_argument('--sample_selection', default=False, type=bool,
                        help='start sample selecion right away?')
    parser.add_argument('--load_config', default=False, type=bool,
                        help='load config from the checkpoint?')
    parser.add_argument('--load_cls_dist', default="", type=str,
                        help='load class distributions path?')
    parser.add_argument('--label_type', default='highest', type=str,
                        help='label new samples based on?')
    args = parser.parse_args()

    config = json.load(open(args.config))
    if args.resume and args.load_config:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args.resume,args.cont,args.sample_selection,args.load_cls_dist)
