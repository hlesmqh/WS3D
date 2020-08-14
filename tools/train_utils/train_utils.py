import logging
import os
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import tqdm
import torch.optim.lr_scheduler as lr_sched
import math
import matplotlib.pyplot as plt
import lib.utils.calibration as calibration
from lib.utils.distance import distance_2
import numpy as np
from lib.config import cfg
from lib.utils.bbox_transform import decode_center_target, decode_bbox_target_stage_2, refine_box
from collections import namedtuple
import lib.utils.iou3d.iou3d_utils as iou3d_utils

logging.getLogger(__name__).addHandler(logging.StreamHandler())
cur_logger = logging.getLogger(__name__)


def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, iter=None):
        if iter is None:
            iter = self.last_iter + 1

        self.last_iter = iter
        self.model.apply(self.setter(self.lmbd(iter)))


class CosineWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


def checkpoint_state(model=None, optimizer=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'it': it, 'model_state': model_state, 'optimizer_state': optim_state}


def save_checkpoint(state, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model=None, optimizer=None, filename='checkpoint', logger=cur_logger):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        it = checkpoint.get('it', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info("==> Done")
    else:
        raise FileNotFoundError

    return it, epoch


def load_part_ckpt(model, filename, logger=cur_logger, total_keys=-1):
    if os.path.isfile(filename):
        logger.info("==> Loading part model from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model_state = checkpoint['model_state']

        update_model_state = {key: val for key, val in model_state.items() if key in model.state_dict()}
        state_dict = model.state_dict()
        state_dict.update(update_model_state)
        model.load_state_dict(state_dict)

        update_keys = update_model_state.keys().__len__()
        if update_keys == 0:
            raise RuntimeError
        logger.info("==> Done (loaded %d/%d)" % (update_keys, total_keys))
    else:
        raise FileNotFoundError


class Trainer(object):
    def __init__(self, model, model_fn, optimizer, ckpt_dir, lr_scheduler, bnm_scheduler,
                 model_fn_eval, tb_log, eval_frequency=1, lr_warmup_scheduler=None, warmup_epoch=-1,
                 grad_norm_clip=1.0):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler, self.model_fn_eval = \
            model, model_fn, optimizer, lr_scheduler, bnm_scheduler, model_fn_eval

        ModelReturn = namedtuple("ModelReturn", ['loss', 'tb_dict', 'disp_dict', "visual_dict"])
        self.MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()

        self.ckpt_dir = ckpt_dir
        self.eval_frequency = eval_frequency
        self.tb_log = tb_log
        self.lr_warmup_scheduler = lr_warmup_scheduler
        self.warmup_epoch = warmup_epoch
        self.grad_norm_clip = grad_norm_clip

    def _train_it(self, batch, prob_mask_ratio):
        self.model.train()

        self.optimizer.zero_grad()
        loss, tb_dict, disp_dict, visual_dict = self.model_fn(self.model, batch, prob_mask_ratio)

        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        return loss.item(), tb_dict, disp_dict, visual_dict

    def eval_epoch_rpn(self, d_loader):
        self.model.eval()

        eval_dict = {}
        total_loss = count = 0.0

        # eval one epoch
        # cls_offset = rpn_cls - rpn_cls_label
        # self.tb_log.add_histogram('cls_offset', cls_offset, it)
        offset_list = []
        offset_x_list = []
        offset_z_list = []
        offset_cls_list = []
        fg_pts_count = 0
        gt_count = 0
        pt_flag = 0
        gt_flag = 0
        for i, data in tqdm.tqdm(enumerate(d_loader, 0), total=len(d_loader), leave=False, desc='val'):
            self.optimizer.zero_grad()

            loss, tb_dict, disp_dict, visual_dict = self.model_fn_eval(self.model, data)
            total_loss += loss.item()
            count += 1

            for k, v in tb_dict.items():
                eval_dict[k] = eval_dict.get(k, 0) + v

            rpn_gt_center = torch.from_numpy(data['gt_centers'][0]).cuda()
            rpn_gt_center = rpn_gt_center[rpn_gt_center.sum(1) > 0]
            rpn_cls = visual_dict['rpn_cls']
            rpn_reg = visual_dict['rpn_reg']
            rpn_xyz = visual_dict['inputs'][:, :3]
            rpn_cls_label = visual_dict['rpn_cls_label'].view(-1, 1)
            rpn_reg_label = visual_dict['rpn_reg_label'].view(-1, 3)

            # classification evaluation
            norm_rpn_cls = torch.sigmoid(rpn_cls).view(-1, 1)
            offset_cls = norm_rpn_cls - rpn_cls_label
            offset_cls_list.append(offset_cls.view(-1, 1))

            # new ppr
            pt_flag += ((norm_rpn_cls > 0.3) & (rpn_cls_label > 0.3)).float().sum()
            fg_pts_count += ((norm_rpn_cls > 0.3)).float().sum()

            # regression evaluation
            fg_mask = norm_rpn_cls.view(-1) > 0.3
            fg_sum = torch.sum(fg_mask)
            # if rpn_gt_center.shape[0] == 0:
            #     fg_pts_count += fg_sum
            if fg_sum != 0 and rpn_gt_center.shape[0] != 0:
                rpn_xyz = rpn_xyz[fg_mask]
                roi_center = torch.zeros_like(rpn_reg[:, :3])
                pred_center = decode_center_target(roi_center[fg_mask].view(-1, roi_center.shape[-1]),
                                                   rpn_reg[fg_mask].view(-1, rpn_reg.shape[-1]),
                                                   loc_scope=cfg.RPN.LOC_SCOPE,
                                                   loc_bin_size=cfg.RPN.LOC_BIN_SIZE).view(-1, 3)
                pred_center += rpn_xyz
                pred_gt_distance = distance_2(rpn_gt_center[:, [0, 2]], pred_center[:, [0, 2]])
                center_min = torch.min(pred_gt_distance, dim=1)[0]
                center_min_index = torch.argmin(pred_gt_distance, dim=1)[center_min.view(-1) < 1.4]
                recall_gt_list = center_min_index.detach().cpu().numpy().tolist()

                #pt_flag += len(recall_gt_list)
                recall_gt_list = list(set(recall_gt_list))
                gt_flag += len(recall_gt_list)
                #fg_pts_count += center_min.shape[0]
                gt_count += rpn_gt_center.shape[0]

                offset_x = torch.min(distance_2(rpn_gt_center[:, [0, 0]], pred_center[:, [0, 0]]) / 0.707, dim=1)[0]
                offset_z = torch.min(distance_2(rpn_gt_center[:, [2, 2]], pred_center[:, [2, 2]]) / 0.707, dim=1)[0]

                valid_index = center_min.view(-1) < 2.0
                if valid_index.long().sum() > 0:
                    offset_list.append(center_min[valid_index])
                    offset_x_list.append(offset_x[valid_index])
                    offset_z_list.append(offset_z[valid_index])

        # statistics this epoch
        for k, v in eval_dict.items():
            eval_dict[k] = eval_dict[k] / max(count, 1)

        if len(offset_list) > 0:
            offset = torch.cat(offset_list, dim=0).reshape(-1, 1)
            offset_x = torch.cat(offset_x_list, dim=0).reshape(-1, 1)
            offset_z = torch.cat(offset_z_list, dim=0).reshape(-1, 1)

            self.tb_log.add_histogram('val_offset_cls', offset_cls[:, 0], self.it)
            self.tb_log.add_histogram('val_offset_x', offset_x, self.it)
            self.tb_log.add_histogram('val_offset_z', offset_z, self.it)
            self.tb_log.add_histogram('val_offset', offset, self.it)

            self.tb_log.add_scalar('val_point_precision', pt_flag / float(fg_pts_count), self.it)
            self.tb_log.add_scalar('val_gt_recall', gt_flag / float(gt_count), self.it)
            print('pprecision: %.2f, grecall: %.2f' % (pt_flag / float(fg_pts_count), gt_flag / float(gt_count)))

        return total_loss / count, eval_dict


    def eval_epoch_rcnn(self, d_loader):
        self.model.eval()

        eval_dict = {'recalled_0.5': 0, 'recalled_0.7': 0}
        total_loss = count = 0.0

        # eval one epoch
        cls=[]
        iou =[]
        p_iou=[]
        iou_offset=[]
        recall05_list = []
        recall07_list = []
        all_gt_list = []
        offset =[]
        for i, data in tqdm.tqdm(enumerate(d_loader, 0), total=len(d_loader), leave=False, desc='val'):
            self.optimizer.zero_grad()

            sample_id = int(data['sample_id'].tolist()[0])
            box_id = int(data['box_id'].tolist()[0])
            if box_id != -1:
                all_gt_list.append([sample_id, box_id])
            if data['cur_box_point'].shape[1]==0: continue
            gt_boxes = torch.from_numpy(data['gt_boxes']).cuda().float()
            loss, tb_dict, disp_dict, visual_dict = self.model_fn_eval(self.model, data)
            total_loss += loss.item()
            #count += 1

            for k, v in tb_dict.items():
                eval_dict[k] = eval_dict.get(k, 0) + v

            rcnn_cls = visual_dict['rcnn_cls']
            cls.append(rcnn_cls)
            cls_label = data['cls'].float().view(-1)
            reg_valid_mask = (cls_label.float()).view(-1)
            fg_mask = reg_valid_mask > 0
            if fg_mask != 0:
                rcnn_reg = visual_dict['rcnn_reg']
                pred_boxes3d = visual_dict['pred_boxes3d']

                _, iou3d = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d.squeeze(1), gt_boxes.squeeze(1))
                eye = torch.from_numpy(np.arange(0, iou3d.shape[0]).reshape(-1, 1)).long().cuda()
                iou3d = torch.gather(iou3d, 1, eye)

                eval_dict['recalled_0.5'] += torch.sum((iou3d > 0.5).long()).item()
                eval_dict['recalled_0.7'] += torch.sum((iou3d > 0.7).long()).item()
                if iou3d > 0.5:
                    recall05_list.append([sample_id,box_id])
                if iou3d > 0.7:
                    recall07_list.append([sample_id,box_id])

                iou.append(iou3d.view(-1,1))
                pred_boxes3d = pred_boxes3d.view(-1,7)
                gt_boxes = gt_boxes[:, :, :7].view(-1,7)
                offset.append(pred_boxes3d[:,:] - gt_boxes[:,:7])
                count += 1


        # statistics this epoch
        for k, v in eval_dict.items():
            eval_dict[k] = eval_dict[k] / max(count, 1)

        cls = torch.cat(cls,dim=0)
        offset = torch.cat(offset,dim=0).reshape(-1,7)
        iou = torch.cat(iou, dim=0).reshape(-1, 1)

        single_gt_list = []
        for sample in all_gt_list:
            if not sample in single_gt_list: single_gt_list.append(sample)
        recall05_list = [sample for sample in single_gt_list if sample in recall05_list]
        recall07_list = [sample for sample in single_gt_list if sample in recall07_list]

        eval_dict['single_recalled_0.5'] = (len(recall05_list) / float(len(single_gt_list)))
        eval_dict['single_recalled_0.7'] = (len(recall07_list) / float(len(single_gt_list)))


        self.tb_log.add_histogram('val_cls', cls, self.it)
        self.tb_log.add_histogram('val_iou', iou.view(-1), self.it)
        self.tb_log.add_histogram('val_x_offset', offset[:, 0], self.it)
        self.tb_log.add_histogram('val_y_offset', offset[:, 1], self.it)
        self.tb_log.add_histogram('val_z_offset', offset[:, 2], self.it)
        self.tb_log.add_histogram('val_h_offset', offset[:, 3], self.it)
        self.tb_log.add_histogram('val_w_offset', offset[:, 4], self.it)
        self.tb_log.add_histogram('val_l_offset', offset[:, 5], self.it)
        self.tb_log.add_histogram('val_ry_offset', offset[:, 6], self.it)

        # if cfg.IOUN.ENABLED:
        #     self.tb_log.add_histogram('val_pred_iou', p_iou.view(-1), self.it)
        #     self.tb_log.add_histogram('val_iou_offset', iou_offset, self.it)

        print('Recall_0.5 %.4f.' % eval_dict['recalled_0.5'])
        print('Recall_0.7 %.4f.' % eval_dict['recalled_0.7'])
        print('Single_Recall_0.5 %.4f.' % eval_dict['single_recalled_0.5'])
        print('Single_Recall_0.7 %.4f.' % eval_dict['single_recalled_0.7'])

        return total_loss / count, eval_dict


    def eval_epoch_ioun(self, d_loader):
        self.model.eval()

        eval_dict = {'recalled_0.5': 0, 'recalled_0.7': 0,
                     'ref_recalled_0.5': 0, 'ref_recalled_0.7': 0}
        total_loss = count = 0.0

        # eval one epoch
        cls = []
        iou = []
        score = []
        p_iou = []
        iou_offset = []
        offset = []
        ref_offset = []
        recall_list = []
        all_gt_list = []
        ref_iou = []
        ref_recall_list = []
        ref_box_list =[]
        TP = 0
        FP = 0
        FN = 0
        for i, data in tqdm.tqdm(enumerate(d_loader, 0), total=len(d_loader), leave=False, desc='val'):
            self.optimizer.zero_grad()

            sample_id = int(data['sample_id'].tolist()[0])
            box_id = int(data['box_id'].tolist()[0])
            if box_id != -1:
                all_gt_list.append([sample_id, box_id])

            if data['cur_box_point'].shape[1]==0: continue
            gt_boxes = torch.from_numpy(data['gt_boxes'][...,:7]).cuda().float()
            loss, tb_dict, disp_dict, visual_dict = self.model_fn_eval(self.model, data)
            total_loss += loss.item()
            #count += 1

            for k, v in tb_dict.items():
                eval_dict[k] = eval_dict.get(k, 0) + v

            cls_label = data['cls'].float().view(-1)
            reg_valid_mask = (cls_label.float()).view(-1)

            fg_mask = reg_valid_mask > 0
            if fg_mask != 0:

                pred_boxes3d = visual_dict['pred_boxes3d'].view(-1, 7)
                refined_boxes3d = visual_dict['refined_box'].view(-1, 7)

                gt_boxes = gt_boxes[:, :, :7].view(-1, 7)

                _, iou3d = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d, gt_boxes)
                eye = torch.from_numpy(np.arange(0, iou3d.shape[0]).reshape(-1, 1)).long().cuda()
                iou3d = torch.gather(iou3d, 1, eye)

                eval_dict['recalled_0.5'] += torch.sum((iou3d > 0.5).long()).item()
                eval_dict['recalled_0.7'] += torch.sum((iou3d > 0.7).long()).item()

                pred_iou = visual_dict['rcnn_iou'] # torch.sigmoid(rcnn_reg[:, -1])
                p_iou.append(pred_iou.view(-1, 1))
                iou_offset.append((iou3d.view(-1,1)-pred_iou.view(-1,1)))
                iou.append(iou3d.view(-1,1))
                offset.append(pred_boxes3d[:,:] - gt_boxes[:,:7])
                count += 1

                recall_list.append([sample_id, box_id, pred_iou.item(), iou3d.item()])



                _, iou3d = iou3d_utils.boxes_iou3d_gpu(refined_boxes3d.view(-1, 7), gt_boxes.view(-1, 7))
                eye = torch.from_numpy(np.arange(0, iou3d.shape[0]).reshape(-1, 1)).long().cuda()
                iou3d = torch.gather(iou3d, 1, eye)

                eval_dict['ref_recalled_0.5'] += torch.sum((iou3d > 0.5).long()).item()
                eval_dict['ref_recalled_0.7'] += torch.sum((iou3d > 0.7).long()).item()

                ref_iou.append(iou3d.view(-1, 1))
                ref_offset.append(refined_boxes3d[:,:] - gt_boxes[:,:7])
                ref_box_list.append(refined_boxes3d[:, :])
                ref_recall_list.append([sample_id, box_id, pred_iou.item(), iou3d.item()])

        # statistics this epoch
        for k, v in eval_dict.items():
            eval_dict[k] = eval_dict[k] / max(count, 1)

        offset = torch.cat(offset,dim=0).reshape(-1,7)
        ref_offset = torch.cat(ref_offset, dim=0).reshape(-1, 7)
        ref_box_list = torch.cat(ref_box_list, dim=0).reshape(-1, 7)

        iou_offset = torch.cat(iou_offset,dim=0).reshape(-1,1)
        p_iou = torch.cat(p_iou, dim=0).reshape(-1, 1)
        iou = torch.cat(iou, dim=0).reshape(-1, 1)
        ref_iou = torch.cat(ref_iou, dim=0).reshape(-1, 1)

        single_gt_list = []
        for sample in all_gt_list:
            if not sample in single_gt_list: single_gt_list.append(sample)

        similar_nms_list = []
        recall_list.sort(key=lambda x: x[2], reverse=True)

        for sample in recall_list:
            exist_flag = False
            for target in similar_nms_list:
                if sample[0:2] == target[0:2]:
                    exist_flag=True
                    if sample[2]>target[2]:
                        target[3] = sample[3]
            if exist_flag == False:
                similar_nms_list.append(sample)

        recall05_list = [sample for sample in similar_nms_list if sample[3]>0.5]
        recall07_list = [sample for sample in similar_nms_list if sample[3]>0.7]

        similar_nms_list = []
        ref_recall_list.sort(key=lambda x: x[2], reverse=True)

        for sample in ref_recall_list:
            exist_flag = False
            for target in similar_nms_list:
                if sample[0:2] == target[0:2]:
                    exist_flag=True
                    if sample[2]>target[2]:
                        target[3] = sample[3]
            if exist_flag == False:
                similar_nms_list.append(sample)

        ref_recall05_list = [sample for sample in similar_nms_list if sample[3]>0.5]
        ref_recall07_list = [sample for sample in similar_nms_list if sample[3]>0.7]

        self.tb_log.add_histogram('val_iou', iou.view(-1), self.it)
        self.tb_log.add_histogram('val_ref_iou', ref_iou.view(-1), self.it)
        self.tb_log.add_histogram('val_x_offset', offset[:, 0], self.it)
        self.tb_log.add_histogram('val_y_offset', offset[:, 1], self.it)
        self.tb_log.add_histogram('val_z_offset', offset[:, 2], self.it)
        self.tb_log.add_histogram('val_h_offset', offset[:, 3], self.it)
        self.tb_log.add_histogram('val_w_offset', offset[:, 4], self.it)
        self.tb_log.add_histogram('val_l_offset', offset[:, 5], self.it)
        self.tb_log.add_histogram('val_ry_offset', offset[:, 6], self.it)

        self.tb_log.add_histogram('val_x_roffset', ref_offset[:, 0], self.it)
        self.tb_log.add_histogram('val_y_roffset', ref_offset[:, 1], self.it)
        self.tb_log.add_histogram('val_z_roffset', ref_offset[:, 2], self.it)
        self.tb_log.add_histogram('val_h_roffset', ref_offset[:, 3], self.it)
        self.tb_log.add_histogram('val_w_roffset', ref_offset[:, 4], self.it)
        self.tb_log.add_histogram('val_l_roffset', ref_offset[:, 5], self.it)
        self.tb_log.add_histogram('val_ry_roffset', ref_offset[:, 6], self.it)

        self.tb_log.add_histogram('val_ref_x', ref_box_list[:, 0], self.it)
        self.tb_log.add_histogram('val_ref_z', ref_box_list[:, 2], self.it)
        self.tb_log.add_histogram('val_ref_y', ref_box_list[:, 1], self.it)
        self.tb_log.add_histogram('val_ref_h', ref_box_list[:, 3], self.it)
        self.tb_log.add_histogram('val_ref_w', ref_box_list[:, 4], self.it)
        self.tb_log.add_histogram('val_ref_l', ref_box_list[:, 5], self.it)
        self.tb_log.add_histogram('val_ref_ry', ref_box_list[:, 6] % (np.pi), self.it)

        self.tb_log.add_histogram('val_pred_iou', p_iou.view(-1), self.it)
        self.tb_log.add_histogram('val_iou_offset', iou_offset, self.it)

        eval_dict['single_recalled_0.5'] = (len(recall05_list) / float(len(single_gt_list)))
        eval_dict['single_recalled_0.7'] = (len(recall07_list) / float(len(single_gt_list)))

        eval_dict['single_ref_recalled_0.5'] = (len(ref_recall05_list) / float(len(single_gt_list)))
        eval_dict['single_ref_recalled_0.7'] = (len(ref_recall07_list) / float(len(single_gt_list)))

        print('Recall_0.5 %.4f.' % eval_dict['recalled_0.5'])
        print('Recall_0.7 %.4f.' % eval_dict['recalled_0.7'])
        print('Recall_ref0.5 %.4f.' % eval_dict['ref_recalled_0.5'])
        print('Recall_ref0.7 %.4f.' % eval_dict['ref_recalled_0.7'])
        return total_loss / count, eval_dict

    def train(self, it, start_iter, total_iters, train_loader, test_loader=None, ckpt_save_interval=5,
              lr_scheduler_each_iter=False):

        start_epoch = int(it / len(train_loader))
        n_epochs = int(total_iters / len(train_loader))
        eval_frequency = self.eval_frequency if self.eval_frequency > 0 else 1
        save_epoch_list = [x for x in range(0, n_epochs, int(n_epochs / min(n_epochs,ckpt_save_interval))) if x>0]+[n_epochs-start_epoch-1]
        eval_epoch_list = [x for x in range(0, n_epochs, int(n_epochs / min(n_epochs,eval_frequency))) ]+[n_epochs-start_epoch-1]
        performance_stop_flag = []
        with tqdm.trange(start_epoch, n_epochs, desc='epochs') as tbar, \
                tqdm.tqdm(total=len(train_loader), leave=False, desc='train') as pbar:

            for epoch in tbar:

                #train one epoch
                for cur_it, batch in enumerate(train_loader):

                    if self.bnm_scheduler is not None:
                        self.bnm_scheduler.step(it)
                        self.tb_log.add_scalar('bn_momentum', self.bnm_scheduler.lmbd(it), it)

                    if lr_scheduler_each_iter:
                        self.lr_scheduler.step(it)
                        cur_lr = float(self.optimizer.lr)
                        self.tb_log.add_scalar('learning_rate', cur_lr, it)
                    else:

                        cur_lr = self.lr_scheduler.get_lr()[0]

                    prob_mask_ratio = 0.5 + 0.5 * (start_epoch + epoch + n_epochs/3) / float(n_epochs)

                    loss, tb_dict, disp_dict, visual_dict = self._train_it(batch, prob_mask_ratio)

                    disp_dict.update({'loss': loss, 'lr': cur_lr})

                    it += 1

                    #visualization rpn output
                    # visualization_rpn_output()

                    # log to console and tensorboard
                    pbar.update()
                    pbar.set_postfix(dict(total_it=it))
                    tbar.set_postfix(disp_dict)
                    tbar.refresh()

                    self.it = it
                    if self.tb_log is not None:

                        if it%10==0:
                            if cfg.RPN.ENABLED:
                                self.rpn_tensorboard(visual_dict, batch, it)
                            elif cfg.RCNN.ENABLED:
                                self.rcnn_tensorboard(visual_dict, batch, it)
                            elif cfg.IOUN.ENABLED:
                                self.ioun_tensorboard(visual_dict, batch, it)
                            else:
                                NotImplementedError

                        self.tb_log.add_scalar('train_loss', loss, it)
                        self.tb_log.add_scalar('learning_rate', cur_lr, it)
                        for key, val in tb_dict.items():
                            self.tb_log.add_scalar('train_' + key, val, it)

                # save trained model
                if epoch in save_epoch_list:
                    ckpt_name = os.path.join(self.ckpt_dir, 'checkpoint_iter_%05d' % it)
                    save_checkpoint(
                        checkpoint_state(self.model, self.optimizer, it), filename=ckpt_name,
                    )

                # eval one epoch
                if epoch in eval_epoch_list:
                    pbar.close()
                    if test_loader is not None:
                        with torch.set_grad_enabled(False):
                            if cfg.RPN.ENABLED:
                                val_loss, eval_dict = self.eval_epoch_rpn(test_loader)
                            elif cfg.RCNN.ENABLED:
                                val_loss, eval_dict = self.eval_epoch_rcnn(test_loader)
                            elif cfg.IOUN.ENABLED:
                                val_loss, eval_dict = self.eval_epoch_ioun(test_loader)
                            else:
                                NotImplementedError
                        if self.tb_log is not None:
                            self.tb_log.add_scalar('val_loss', val_loss, it)
                            for key, val in eval_dict.items():
                                self.tb_log.add_scalar('val_' + key, val, it)

                pbar.close()
                pbar = tqdm.tqdm(total=len(train_loader), leave=False, desc='train')
                pbar.set_postfix(dict(total_it=it))



        return None

    def rpn_tensorboard(self, visual_dict, batch, it):
        # only visualize the first instance
        # todo: GT recall is not correctly defined,only in eval_rpn is right
        rpn_gt_center = torch.from_numpy(batch['gt_centers'][0]).cuda()
        rpn_gt_center = rpn_gt_center[rpn_gt_center.sum(1) > 0]
        rpn_cls = visual_dict['rpn_cls']
        rpn_reg = visual_dict['rpn_reg']
        rpn_xyz = visual_dict['inputs'][:, :3]
        rpn_cls_label = visual_dict['rpn_cls_label'].view(-1, 1)
        rpn_reg_label = visual_dict['rpn_reg_label'].view(-1, 3)


        # classification evaluation
        norm_rpn_cls = torch.sigmoid(rpn_cls)
        offset_cls = norm_rpn_cls - rpn_cls_label
        self.tb_log.add_histogram('offset_cls', offset_cls, it)

        # regression evaluation
        fg_mask = (norm_rpn_cls > 0.3).view(-1)
        fg_sum = torch.sum(fg_mask)
        if fg_sum != 0:
            rpn_xyz = rpn_xyz[fg_mask]
            roi_center = torch.zeros_like(rpn_reg[:, :3])
            pred_center = decode_center_target(roi_center[fg_mask], rpn_reg[fg_mask].view(-1, rpn_reg.shape[-1]),
                                               loc_scope=cfg.RPN.LOC_SCOPE,
                                               loc_bin_size=cfg.RPN.LOC_BIN_SIZE).view(-1, 3)
            pred_center += rpn_xyz
            pred_gt_distance = distance_2(rpn_gt_center[:, [0, 2]], pred_center[:, [0, 2]])
            center_min = torch.min(pred_gt_distance, dim=1)[0]
            center_min_index = torch.argmin(pred_gt_distance, dim=1)[center_min.view(-1) < 1.4]
            recall_gt_list = center_min_index.detach().cpu().numpy().tolist()

            pt_flag = len(recall_gt_list)
            recall_gt_list = list(set(recall_gt_list))
            gt_flag = len(recall_gt_list)

            valid_index = center_min.view(-1) < 2.0
            if valid_index.long().sum() > 0:
                offset = center_min[valid_index]
                offset_x = torch.min(distance_2(rpn_gt_center[:, [0, 0]], pred_center[:, [0, 0]]) / 0.707, dim=1)[0]
                offset_z = torch.min(distance_2(rpn_gt_center[:, [2, 2]], pred_center[:, [2, 2]]) / 0.707, dim=1)[0]
                self.tb_log.add_histogram('offset', offset, self.it)
                self.tb_log.add_histogram('x_offset', offset_x[valid_index], self.it)
                self.tb_log.add_histogram('z_offset', offset_z[valid_index], self.it)

            PointsTP_num = pt_flag
            GTTP_num = gt_flag
            PointsFP_num = center_min.shape[0] - pt_flag
            GTFN_num = rpn_gt_center.shape[0] - gt_flag
            point_precision = PointsTP_num / float(PointsTP_num + PointsFP_num)
            gt_recall = GTTP_num / float(GTTP_num + GTFN_num)

            self.tb_log.add_scalar('point_precision', point_precision, it)
            self.tb_log.add_scalar('gt_recall', gt_recall, it)




    def rcnn_tensorboard(self, visual_dict, batch, it):

        rcnn_cls = visual_dict['rcnn_cls']
        rcnn_reg = visual_dict['rcnn_reg']
        pred_boxes3d = visual_dict['pred_boxes3d']
        rcnn_cls_label = batch['cls']

        valid_mask = (rcnn_cls_label).view(-1)>0
        rcnn_reg = rcnn_reg[valid_mask]
        gt_boxes=batch['gt_boxes'][valid_mask]
        pred_boxes3d = pred_boxes3d[valid_mask]

        self.tb_log.add_histogram('x_label', gt_boxes[:, 0, 0], it)
        self.tb_log.add_histogram('z_label', gt_boxes[:, 0, 2], it)
        self.tb_log.add_histogram('y_label', gt_boxes[:, 0, 1], it)
        self.tb_log.add_histogram('h_label', gt_boxes[:, 0, 3], it)
        self.tb_log.add_histogram('w_label', gt_boxes[:, 0, 4], it)
        self.tb_log.add_histogram('l_label', gt_boxes[:, 0, 5], it)
        self.tb_log.add_histogram('ry_label', gt_boxes[:, 0, 6] % np.pi, it)

        pred_boxes3d = pred_boxes3d.view(-1, 1, 7)

        iou2d, iou3d = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d.squeeze(1), gt_boxes.squeeze(1))
        eye = torch.from_numpy(np.arange(0, iou3d.shape[0]).reshape(-1, 1)).long().cuda()
        iou2d = torch.gather(iou2d, 1, eye)
        iou3d = torch.gather(iou3d, 1, eye)

        self.tb_log.add_histogram('trans_x', pred_boxes3d[:, :, 0], it)
        self.tb_log.add_histogram('trans_z', pred_boxes3d[:, :, 2], it)
        self.tb_log.add_histogram('trans_y', pred_boxes3d[:, :, 1], it)
        self.tb_log.add_histogram('trans_h', pred_boxes3d[:, :, 3], it)
        self.tb_log.add_histogram('trans_w', pred_boxes3d[:, :, 4], it)
        self.tb_log.add_histogram('trans_l', pred_boxes3d[:, :, 5], it)
        self.tb_log.add_histogram('trans_ry', pred_boxes3d[:, :, 6] % (np.pi), it)

        self.tb_log.add_histogram('x_offset', pred_boxes3d[:, 0, 0] - gt_boxes[:, 0, 0], it)
        self.tb_log.add_histogram('z_offset', pred_boxes3d[:, 0, 2] - gt_boxes[:, 0, 2], it)
        self.tb_log.add_histogram('y_offset', pred_boxes3d[:, 0, 1] - gt_boxes[:, 0, 1], it)
        self.tb_log.add_histogram('h_offset', pred_boxes3d[:, 0, 3] - gt_boxes[:, 0, 3], it)
        self.tb_log.add_histogram('w_offset', pred_boxes3d[:, 0, 4] - gt_boxes[:, 0, 4], it)
        self.tb_log.add_histogram('l_offset', pred_boxes3d[:, 0, 5] - gt_boxes[:, 0, 5], it)
        self.tb_log.add_histogram('ry_offset', ((pred_boxes3d[:, 0, 6] % (np.pi * 2) - gt_boxes[:, 0, 6] % (np.pi * 2))) , it)


        self.tb_log.add_histogram('iou2d', iou2d, it)
        self.tb_log.add_histogram('iou3d', iou3d, it)

        self.tb_log.add_scalar('recalled_0.5', torch.sum((iou3d > 0.5).float()/(rcnn_cls_label).sum()), it)
        self.tb_log.add_scalar('recalled_0.7', torch.sum((iou3d > 0.7).float()/(rcnn_cls_label).sum()), it)

    def ioun_tensorboard(self, visual_dict, batch, it):


        rcnn_cls_label = batch['cls']
        valid_mask = (rcnn_cls_label).view(-1) > 0
        pred_iou=visual_dict['rcnn_iou'][valid_mask]
        rcnn_ref = visual_dict['rcnn_ref'][valid_mask]
        gt_boxes = batch['gt_boxes'][valid_mask]
        pred_boxes3d = visual_dict['pred_boxes3d'][valid_mask]

        iou2d, iou3d = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d.view(-1, 7), gt_boxes.view(-1, 7))
        eye = torch.from_numpy(np.arange(0, iou3d.shape[0]).reshape(-1, 1)).long().cuda()
        iou2d = torch.gather(iou2d, 1, eye)
        iou3d = torch.gather(iou3d, 1, eye)

        self.tb_log.add_scalar('recalled_0.5', torch.sum((iou3d > 0.5).float() / (valid_mask).float().sum()), it)
        self.tb_log.add_scalar('recalled_0.7', torch.sum((iou3d > 0.7).float() / (valid_mask).float().sum()), it)
        self.tb_log.add_histogram('pred_iou', pred_iou.view(-1), it)
        self.tb_log.add_histogram('offset_iou', iou3d.view(-1) - pred_iou.view(-1), it)
        self.tb_log.add_histogram('iou', iou3d.view(-1), it)

        # basic ref
        refine_boxes3d = refine_box(pred_boxes3d, rcnn_ref)

        iou2d, iou3d = iou3d_utils.boxes_iou3d_gpu(refine_boxes3d.view(-1, 7), gt_boxes.view(-1, 7))
        eye = torch.from_numpy(np.arange(0, iou3d.shape[0]).reshape(-1, 1)).long().cuda()
        iou2d = torch.gather(iou2d, 1, eye)
        iou3d = torch.gather(iou3d, 1, eye)

        self.tb_log.add_scalar('ref_recalled_0.5', torch.sum((iou3d > 0.5).float() / (valid_mask).float().sum()), it)
        self.tb_log.add_scalar('ref_recalled_0.7', torch.sum((iou3d > 0.7).float() / (valid_mask).float().sum()), it)
        self.tb_log.add_histogram('ref_pred_iou', pred_iou.view(-1), it)
        self.tb_log.add_histogram('ref_offset_iou', iou3d.view(-1) - pred_iou.view(-1), it)
        self.tb_log.add_histogram('ref_iou', iou3d.view(-1), it)



def visualization_rpn_output():
    return
    # if it%2000 ==1:
    #     def get_image(idx):
    #         import cv2
    #         img_file = os.path.join(train_loader.dataset.image_dir, '%06d.png' % idx)
    #         assert os.path.exists(img_file)
    #         return cv2.imread(img_file)  # (H, W, 3) BGR mode
    #     def get_calib(idx):
    #         calib_file = os.path.join(train_loader.dataset.calib_dir, '%06d.txt' % idx)
    #         assert os.path.exists(calib_file)
    #         return calibration.Calibration(calib_file)
    #     img = get_image(batch['sample_id'][0])
    #     calib = get_calib(batch['sample_id'][0])
    #     inputs = visual_dict['inputs']
    #     output_cls, output_reg = visual_dict['rpn_cls'], visual_dict['rpn_reg']
    #     inputs = inputs.cpu().numpy()
    #     output_cls = 1/(1+np.exp(output_cls.detach().cpu().numpy().reshape(-1) + 1e-8))
    #     output_reg = output_reg.detach().cpu().numpy().reshape(-1,68)
    #     aug_pts_img, _ = calib.rect_to_img(inputs[:,:3])
    #     fig = plt.figure(figsize=(15,5))
    #     plt.title('class')
    #     plt.scatter(aug_pts_img[:,0], aug_pts_img[:,1], s=15, c=output_cls, edgecolor='none', cmap=plt.get_cmap('rainbow'), alpha=1, marker='.', vmin=0, vmax=1)
    #     plt.imshow(img[:,:,[2,1,0]])
    #     plt.axis('off')
    #     plt.show()