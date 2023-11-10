import torch
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from base import BaseTrainer, DataPrefetcher
from utils.helpers import colorize_mask,create_lossy_labels
from utils.metrics import eval_metrics, AverageMeter
from utils.losses import Binary_CELoss
from tqdm import tqdm
import copy
from utils.losses import comp_thresh,mse_loss
import gc


class Trainer(BaseTrainer):
    def __init__(self, model,step, loss, loss2,resume, config, train_loader,train_loader2, train_loader3,val_loader=None,val_loader_2=None, train_logger=None, prefetch=True,thresh=0.5,stage=1):
        super(Trainer, self).__init__(model,step, loss,loss2, resume, config, train_loader,train_loader2, train_loader3,val_loader,val_loader_2, train_logger,stage)

        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        if self.device ==  torch.device('cpu'): prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)
            self.val_loader_2 = DataPrefetcher(val_loader_2, device=self.device)
            self.train_loader2=DataPrefetcher(train_loader2, device=self.device)
            self.train_loader3=DataPrefetcher(train_loader3,device=self.device)
        torch.backends.cudnn.benchmark = True
        self.criterion = Binary_CELoss()


    def _train_epoch_1(self, epoch):
        self.logger.info('\n')

        self.model.train()
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.freeze_everything_but_lossy_Detector(True)
        else:
            self.model.freeze_everything_but_lossy_Detector(True)

        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, target,_) in enumerate(tbar):
            self.data_time.update(time.time() - tic)

            output = self.model(data,1)
            loss_final_seg = self.loss(output, target)

            loss_total=loss_final_seg
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            self.total_loss.update(loss_total.item())
            self.final_seg_loss.update(loss_final_seg.mean().item())
            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar(f'{self.wrt_mode}/loss', loss_total.item(), self.wrt_step)


            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = self._get_seg_metrics().values()

            # PRINT INFO
            tbar.set_description('TRAIN stage1 ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} | B {:.2f} D {:.2f} |'.format(
                                                epoch, self.total_loss.average,
                                                pixAcc, mIoU,
                                                self.batch_time.average, self.data_time.average))
            # METRICS TO TENSORBOARD
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]:
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
            for i, opt_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(f'{self.wrt_mode}/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
            log = {'loss': self.total_loss.average, 'loss_final_seg': self.final_seg_loss.average,
                   **seg_metrics}
            self.lr_scheduler.step(epoch)
        torch.cuda.empty_cache()
        gc.collect()
        return log

    def _train_epoch_2(self, epoch):
        self.logger.info('\n')
        self.model.train()
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.freeze_everything_but_lossy_Detector()
        else:
            self.model.freeze_everything_but_lossy_Detector()

        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader2, ncols=130)

        for batch_idx, (data, target, edges, _) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            output,lossy,middle1,middle2,middle3= self.model(data,2)
            loss_final_seg = self.loss2(output, target)
            lossy_gt = create_lossy_labels(target, loss_final_seg.data.detach(), self.thresh)
            lossy_gt_edge=copy.deepcopy(lossy_gt)
            lossy_gt_edge[edges==0]=255
            lossy_gt_center = copy.deepcopy(lossy_gt)
            lossy_gt_center[edges == 1] = 255
            area_loss = self.criterion(lossy[:,0:1,:,:].float(), lossy_gt_edge.unsqueeze(1))+self.criterion(lossy[:,1:2,:,:].float(), lossy_gt_center.unsqueeze(1))
            loss_middle = 0.1 * self.loss(middle1, target) + 0.2 * self.loss(middle2, target) + 0.3 * self.loss(middle3,
                                                                                                               target)
            loss_total = area_loss+0.5*loss_middle

            self.optimizer2.zero_grad()
            loss_total.backward()
            self.optimizer2.step()

            self.total_loss.update(loss_total.item())
            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar(f'{self.wrt_mode}/loss', loss_total.item(), self.wrt_step)
            tbar.set_description('TRAIN stage2 ({}) | Loss: {:.3f} | B {:.2f} D {:.2f} |'.format(
                epoch, self.total_loss.average,
                self.batch_time.average, self.data_time.average))
            log = {'loss': self.total_loss.average}
            self.lr_scheduler2.step(epoch-self.start_epoch)
        torch.cuda.empty_cache()
        gc.collect()
        return log

    def get_weights(self):
        tbar = tqdm(self.train_loader3, ncols=130)
        self.weights=torch.zeros(self.num_classes,device=self.device)
        Counts=torch.zeros(self.num_classes,device=self.device)
        for batch_idx, (_,target,_, _) in enumerate(tbar):
            existing_classes,counts = torch.unique(target,return_counts=True)
            counts = counts[existing_classes != 255]
            existing_classes = existing_classes[existing_classes != 255]
            Counts[existing_classes]+=counts
        self.weights=1-Counts/Counts.sum()




    def get_thresh(self):
        self.model.eval()
        tbar = tqdm(self.train_loader3, ncols=130)
        self.thresh = torch.zeros(self.num_classes)
        Counts = torch.zeros(self.num_classes)
        with torch.no_grad():
            for (data,target,_,_) in tbar:
                output = self.model(data,1)
                loss = self.loss2(output,target)
                thresh, count = comp_thresh(loss, target, 19)
                self.thresh += thresh
                Counts += count

        self.thresh /= Counts
        torch.cuda.empty_cache()
        gc.collect()

    def _valid_epoch_1(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data,target,_) in enumerate(tbar):
                output=self.model(data,1)
                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( epoch,
                                                self.total_loss.average,
                                                pixAcc, mIoU))

                del loss,output

            # WRTING & VISUALIZING THE MASKS
            val_img = []
            palette = self.train_loader.dataset.palette
            for d, t, o in val_visual:
                d = self.restore_transform(d)
                t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                val_img.extend([d, t, o])
            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
            self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]:
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }
        return log

    def _valid_epoch_2(self, epoch):
        if self.val_loader_2 is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader_2, ncols=130)
        with torch.no_grad():
            for batch_idx, (data,target,edges) in enumerate(tbar):

                output,lossy_area,_,_,_=self.model(data,2)
                loss_final_seg = self.loss2(output, target)
                lossy_gt=create_lossy_labels(target,loss_final_seg.data.detach(),self.thresh)
                lossy_gt_edges=copy.deepcopy(lossy_gt)
                lossy_gt_edges[edges==0]=255
                lossy_gt_centers = copy.deepcopy(lossy_gt)
                lossy_gt_centers[edges == 1] = 255

                loss = self.criterion(lossy_area[:,0:1,:,:].float(),lossy_gt_edges.unsqueeze(1))+self.criterion(lossy_area[:,1:2,:,:].float(),lossy_gt_centers.unsqueeze(1))

                self.total_loss.update(loss.item())
                tbar.set_description('EVAL stage 2 ({}) | Loss: {:.3f}'.format(epoch,self.total_loss.average))
                del lossy_area,lossy_gt,loss_final_seg
                del loss,output

            log = {
                'MSE': self.total_loss.average
            }


        return log


    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.final_seg_loss=AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }
