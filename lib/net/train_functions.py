import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.utils.loss_utils as loss_utils
import torch.autograd as grad
from lib.config import cfg
from collections import namedtuple
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import lib.utils.kitti_utils as kitti_utils
from lib.utils.kitti_utils import boxes3d_to_corners3d_torch
from lib.utils.bbox_transform import decode_center_target, decode_bbox_target_stage_2
import lib.utils.iou3d.iou3d_utils as iou3d_utils
from lib.utils.giou_utils import gious_3d_loss, ious_3d_loss

def model_joint_fn_decorator():
    ModelReturn = namedtuple("ModelReturn", ['loss', 'tb_dict', 'disp_dict', "visual_dict"])
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
    object_gious_3d_loss = gious_3d_loss()
    object_ious_3d_loss = ious_3d_loss()

    def model_fn(model, data, prob_mask_ratio=1.0):
        if cfg.RPN.ENABLED:
            pts_input = data['pts_input']
            gt_centers = data['gt_centers']

            rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
            if cfg.RPN.Gaussian_Center:
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking=True).float()
            else:
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking=True).long()
            rpn_reg_label = torch.from_numpy(rpn_reg_label).cuda(non_blocking=True).float()

            inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
            gt_boxes3d = torch.from_numpy(gt_centers).cuda(non_blocking=True).float()
            input_data = {'pts_input': inputs, 'gt_centers': gt_centers}

        elif cfg.RCNN.ENABLED or cfg.IOUN.ENABLED:
            input_data = data
            for key, val in data.items():
                if not key in ['sample_id', 'box_id']:
                    input_data[key] = torch.from_numpy(val).contiguous().cuda(non_blocking=True).float()

            if random.random() > prob_mask_ratio:
                input_data['train_mask'] = data['gt_mask'].float()
            else:
                input_data['train_mask'] = data['cur_prob_mask'].float()

            ext_noise, revive_matrix = input_data['ext_noise'].reshape(-1, 1, 3), input_data['revive_matrix']
            Rot_y, noise_scale = input_data['Rot_y'], input_data['noise_scale']
            cur_box_point, gt_boxes = input_data['cur_box_point'], input_data['gt_boxes']

            # extra_size noise
            cur_box_point = torch.einsum('ijk,ikl->ijl', cur_box_point, revive_matrix[:, 0, ...].permute(0, 2, 1))
            cur_box_point[:, :, 0:3] = torch.mul(cur_box_point[:, :, 0:3], ext_noise[:, :, [1, 0, 2]])
            gt_boxes[:, :, 3:6] = torch.mul(gt_boxes[:, :, 3:6], ext_noise)
            cur_box_point = torch.einsum('ijk,ikl->ijl', cur_box_point, revive_matrix[:, 1, ...].permute(0, 2, 1))

            cur_box_point[:,:,0:3] = torch.mul(cur_box_point[:,:,0:3], noise_scale.repeat(1,cur_box_point.shape[1],3))
            cur_box_point = torch.einsum('ijk,ikl->ijl', cur_box_point, Rot_y.permute(0,2,1))[:, :, 0:3]
            input_data['cur_box_point'] = cur_box_point.contiguous()

            gt_boxes[:, :, 0:6] = torch.mul(gt_boxes[:, :, 0:6], noise_scale.repeat(1,1,6))
            gt_boxes[:,:,0:3] = torch.einsum('ijk,ikl->ijl', gt_boxes[:,:,[0,1,2,7]], Rot_y.permute(0, 2, 1))[:, :, 0:3]
            gt_boxes = gt_boxes[:,:,0:7]
            input_data['gt_boxes'] = gt_boxes.contiguous()

        else:
            NotImplementedError

        # cur_box_point_n = cur_box_point.detach().cpu().numpy()[1,:,:]
        # gt_boxes_n = gt_boxes.detach().cpu().numpy()[1,0,:]
        # fig, ax = plt.subplots(figsize=(5, 5))
        # ax.axis([-4, 4, -4, 4])
        # plt.scatter(cur_box_point_n[:, 0], cur_box_point_n[:, 2], s=15, c=cur_box_point_n[:,1], edgecolor='none',
        #             cmap=plt.get_cmap('rainbow'), alpha=1, marker='.', vmin=-1, vmax=1)
        # plt.scatter(np.zeros(1), np.zeros(1), s=200, c='black',
        #             alpha=0.5, marker='x', vmin=-1, vmax=1)
        # plt.scatter(gt_boxes_n[0], gt_boxes_n[2], s=200, c='blue',
        #             alpha=0.5, marker='+', vmin=-1, vmax=1)
        # gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes_n.reshape(-1, 7), rotate=True)
        # pred_boxes3d_corner = gt_corners
        # print_box_corner = pred_boxes3d_corner[0]
        # x1, x2, x3, x4 = print_box_corner[0:4, 0]
        # z1, z2, z3, z4 = print_box_corner[0:4, 2]
        # polygon = np.zeros([5, 2], dtype=np.float32)
        # polygon[0, 0] = x1
        # polygon[1, 0] = x2
        # polygon[2, 0] = x3
        # polygon[3, 0] = x4
        # polygon[4, 0] = x1
        # polygon[0, 1] = z1
        # polygon[1, 1] = z2
        # polygon[2, 1] = z3
        # polygon[3, 1] = z4
        # polygon[4, 1] = z1
        # line1 = [(x1, z1), (x2, z2)]
        # line2 = [(x2, z2), (x3, z3)]
        # line3 = [(x3, z3), (x4, z4)]
        # line4 = [(x4, z4), (x1, z1)]
        # (line1_xs, line1_ys) = zip(*line1)
        # (line2_xs, line2_ys) = zip(*line2)
        # (line3_xs, line3_ys) = zip(*line3)
        # (line4_xs, line4_ys) = zip(*line4)
        # ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='green'))
        # ax.add_line(Line2D(line2_xs, line2_ys, linewidth=1, color='red'))
        # ax.add_line(Line2D(line3_xs, line3_ys, linewidth=1, color='red'))
        # ax.add_line(Line2D(line4_xs, line4_ys, linewidth=1, color='red'))
        # plt.show()


        ret_dict = model(input_data)


        tb_dict = {}
        disp_dict = {}
        visual_dict = {}
        loss = 0
        if cfg.RPN.ENABLED:
            rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']
            rpn_loss = get_rpn_loss(model, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict)
            loss += rpn_loss
            disp_dict['cls'] = tb_dict['rpn_loss_cls']
            disp_dict['reg'] = tb_dict['rpn_loss_reg']
            visual_dict['inputs'] = inputs[0]
            visual_dict['rpn_cls'] = rpn_cls[0]
            visual_dict['rpn_reg'] = rpn_reg[0]
            visual_dict['rpn_cls_label'] = rpn_cls_label[0]
            visual_dict['rpn_reg_label'] = rpn_reg_label[0]

        elif cfg.RCNN.ENABLED:
            rcnn_loss = get_rcnn_loss(model, ret_dict, tb_dict, visual_dict)
            # disp_dict['reg_fg_sum'] = tb_dict['rcnn_reg_fg']
            loss += rcnn_loss
            disp_dict['cls'] = tb_dict['rcnn_loss_cls']

            disp_dict['loc'] = tb_dict['rcnn_loss_loc']
            disp_dict['ang'] = tb_dict['rcnn_loss_angle']
            disp_dict['siz'] = tb_dict['rcnn_loss_size']
            disp_dict['cor'] = tb_dict['rcnn_loss_corner']
            disp_dict['giou'] = tb_dict['rcnn_loss_giou']

        elif cfg.IOUN.ENABLED:
            ioun_loss = get_ioun_loss(model, ret_dict, tb_dict, visual_dict, input_data)
            loss += ioun_loss
            disp_dict['iou'] = tb_dict['loss_iou']
            disp_dict['loc'] = tb_dict['ioun_loss_loc']
            disp_dict['ang'] = tb_dict['ioun_loss_ang']
            disp_dict['siz'] = tb_dict['ioun_loss_siz']
            #disp_dict['ref'] = tb_dict['ioun_loss_ref']

        else:
            NotImplementedError

        disp_dict['loss'] = loss.item()

        return ModelReturn(loss, tb_dict, disp_dict, visual_dict)



    def get_rpn_loss(model, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict):
        if isinstance(model, nn.DataParallel):
            rpn_cls_loss_func = model.module.rpn.rpn_cls_loss_func
        else:
            rpn_cls_loss_func = model.rpn.rpn_cls_loss_func

        rpn_cls_label_flat = rpn_cls_label.view(-1)
        rpn_cls_flat = rpn_cls.view(-1)
        # todo: whether to change >0.5 since we use soft label
        fg_mask = (rpn_cls_label_flat > 0)

        # RPN classification loss
        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls, rpn_cls_label_flat)

        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            if cfg.RPN.Gaussian_Center:
                rpn_cls_target = rpn_cls_label_flat.float()
                pos = rpn_cls_label_flat.float()
                neg = (1-rpn_cls_label_flat).float()

            else:
                rpn_cls_target = (rpn_cls_label_flat > 0.5).float()
                pos = (rpn_cls_label_flat > 0.5).float()
                neg = (rpn_cls_label_flat < 0.5).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls_flat, rpn_cls_target, cls_weights)
            rpn_loss_cls_pos = (rpn_loss_cls * pos).sum()
            rpn_loss_cls_neg = (rpn_loss_cls * neg).sum()
            rpn_loss_cls = rpn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rpn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rpn_loss_cls_neg.item()

        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            weight = rpn_cls_flat.new(rpn_cls_flat.shape[0]).fill_(1.0)
            weight[fg_mask] = cfg.RPN.FG_WEIGHT
            rpn_cls_label_target = (rpn_cls_label_flat > 0.5).float()
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rpn_cls_flat), rpn_cls_label_target,
                                                    weight=weight, reduction='none')
            cls_valid_mask = (rpn_cls_label_flat >= 0.5).float()
            rpn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        # RPN regression loss
        point_num = rpn_reg.size(0) * rpn_reg.size(1)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:
            loss_loc, rpn_reg_loss_dict = \
                loss_utils.get_rpn_reg_loss(rpn_reg.view(point_num, -1)[fg_mask],
                                        rpn_reg_label.view(point_num, 3)[fg_mask],
                                        loc_scope=cfg.RPN.LOC_SCOPE,
                                        loc_bin_size=cfg.RPN.LOC_BIN_SIZE)

            rpn_loss_reg = loss_loc
        else:
            rpn_loss_reg = rpn_loss_cls * 0

        rpn_loss = rpn_loss_cls * cfg.RPN.LOSS_WEIGHT[0] + rpn_loss_reg * cfg.RPN.LOSS_WEIGHT[1]

        tb_dict.update({'rpn_loss_cls': rpn_loss_cls.item(), 'rpn_loss_reg': rpn_loss_reg.item(),
                        'rpn_loss': rpn_loss.item(), 'rpn_fg_sum': fg_sum})

        return rpn_loss

    def get_rcnn_loss(model, ret_dict, tb_dict, visual_dict):
        rcnn_cls, rcnn_reg = ret_dict['rcnn_cls'], ret_dict['rcnn_reg']
        batch_size = rcnn_reg.shape[0]

        gt_boxes3d = ret_dict['gt_boxes'].clone().view(batch_size, 7)
        cls_label = ret_dict['cls'].float().view(-1)
        reg_valid_mask = (ret_dict['cls'].float()).view(-1)
        pred_boxes3d = ret_dict['pred_boxes3d'].clone().view(-1, 7)

        # rcnn regression loss
        fg_mask = reg_valid_mask > 0
        fg_sum = torch.sum(fg_mask)
        if fg_sum != 0:
            # rcnn regression
            anchor_size = MEAN_SIZE

            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                loss_utils.get_rcnn_reg_loss(rcnn_reg.view(batch_size, -1)[fg_mask],
                                             gt_boxes3d.view(batch_size, 7)[fg_mask],
                                             loc_scope=cfg.RCNN.LOC_SCOPE,
                                             loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                             num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                             anchor_size=anchor_size,
                                             get_xz_fine=cfg.RCNN.LOC_XZ_FINE, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                             loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                             get_ry_fine=False)

            # extra box loss
            iou2d, iou3d = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d[fg_mask], gt_boxes3d.view(batch_size, 7)[fg_mask])
            eye = torch.from_numpy(np.arange(0, iou3d.shape[0]).reshape(-1, 1)).long().cuda()
            iou3d = torch.gather(iou3d, 1, eye).detach()
            iou_mask = iou3d.view(-1) > 0.5
            iou_sum = torch.sum(iou_mask)
            if iou_sum != 0:
                # corner loss
                gt_boxes3d_fcorner = gt_boxes3d.clone().view(batch_size, 7)[fg_mask][iou_mask]
                pred_corner = boxes3d_to_corners3d_torch(pred_boxes3d[fg_mask][iou_mask])
                gt_corner = boxes3d_to_corners3d_torch(gt_boxes3d_fcorner)
                gt_boxes3d_fcorner[:, 6] += np.pi
                gt_flip_corner = boxes3d_to_corners3d_torch(gt_boxes3d_fcorner)
                corner_dist = torch.min(torch.norm(pred_corner - gt_corner, dim=-1),
                                        torch.norm(pred_corner - gt_flip_corner, dim=-1))
                corner_loss = F.smooth_l1_loss(corner_dist,
                                               torch.zeros_like(corner_dist))
                # corner_loss = torch.zeros(()).cuda()

                # giou loss
                gt_boxes3d_fgiou = gt_boxes3d.clone().view(batch_size, 7)[fg_mask]
                # gious_loss = object_gious_3d_loss(gt_boxes3d_fgiou[iou_mask], pred_boxes3d[iou_mask])
                gious_loss = object_ious_3d_loss(gt_boxes3d_fgiou[iou_mask], pred_boxes3d[fg_mask][iou_mask])
                # gious_loss = torch.zeros(()).cuda()

            else:
                corner_loss = torch.zeros(()).cuda()
                gious_loss = torch.zeros(()).cuda()

            # iou loss calculate
            iou2d, iou3d = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d[fg_mask], gt_boxes3d[fg_mask])
            eye = torch.from_numpy(np.arange(0, iou3d.shape[0]).reshape(-1, 1)).long().cuda()
            iou3d = torch.gather(iou3d, 1, eye).detach()
            iou3d_label = iou3d
            iou3d_label = iou3d_label.pow(2)

            loss_loc = loss_loc * 20
            loss_angle = loss_angle
            loss_size = loss_size * 300
            corner_loss = corner_loss * 10
            rcnn_loss_reg = loss_loc + loss_angle + loss_size  # + reg_error_T1 + reg_error_T2
            reg_loss_dict['loss_loc'] = loss_loc
            reg_loss_dict['loss_angle'] = loss_angle
            reg_loss_dict['loss_size'] = loss_size
            reg_loss_dict['loss_corner'] = corner_loss
            reg_loss_dict['loss_giou'] = gious_loss


        else:
            loss_loc = torch.zeros(()).cuda()
            loss_angle = torch.zeros(()).cuda()
            loss_size = torch.zeros(()).cuda()
            rcnn_loss_reg = torch.zeros(()).cuda()
            corner_loss = torch.zeros(()).cuda()
            gious_loss = torch.zeros(()).cuda()

        # rcnn classification loss
        if isinstance(model, nn.DataParallel):
            cls_loss_func = model.module.rcnn_net.cls_loss_func
        else:
            cls_loss_func = model.rcnn_net.cls_loss_func

        cls_label_flat = cls_label.view(-1)

        if cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            invalid_mask = gt_boxes3d.sum(-1) != 0
            # BCE
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), cls_label, reduction='none')
            cls_valid_mask = (cls_label_flat >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

            # BalanceBCE
            # rcnn_cls_flat = rcnn_cls.view(-1)
            # batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), cls_label, reduction='none')
            # cls_valid_mask = ((cls_label_flat >= 0)&(~invalid_mask)).float()
            # rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)\
            #                 +(batch_loss_cls * (1-cls_valid_mask)).sum() / torch.clamp((1-cls_valid_mask).sum(), min=1.0)

            # BCEIOU
            # rcnn_cls_flat = rcnn_cls.view(-1)
            # if fg_sum != 0:
            #     cls_label[fg_mask]=iou3d_label.view(-1)
            # batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), cls_label, reduction='none')
            # cls_valid_mask = (cls_label_flat >= 0).float()
            # rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

            # MSE
            # if fg_sum != 0:
            #     rcnn_loss_cls = F.mse_loss(rcnn_cls.view(-1)[fg_mask], iou3d_label.view(-1)) * 100
            # else:
            #     rcnn_loss_cls = torch.zeros(()).cuda()

            # GMSE
            # cls_label = cls_label*0
            # if fg_sum != 0:
            #     cls_label[fg_mask]=iou3d_label.view(-1)
            # rcnn_loss_cls = F.mse_loss(rcnn_cls.view(-1), cls_label.view(-1)) * 100

            # BalancedGMSE
            # cls_label = cls_label*0
            # if fg_sum != 0:
            #     cls_label[fg_mask]=iou3d_label.view(-1)
            # batch_loss_cls = F.mse_loss(rcnn_cls.view(-1), cls_label.view(-1), reduction='none')
            # rcnn_loss_cls = (batch_loss_cls * fg_mask.float()).sum() / torch.clamp(fg_mask.float().sum(), min=1.0) \
            #                 + (batch_loss_cls * (~fg_mask).float()).sum() / torch.clamp((~fg_mask).float().sum(), min=1.0)
            #
            # rcnn_loss_cls = rcnn_loss_cls * 100
        else:
            raise NotImplementedError

        # rcnn training
        if cfg.RCNN.ENABLED:
            rcnn_loss = rcnn_loss_cls + rcnn_loss_reg + corner_loss

        tb_dict['rcnn_loss_cls'] = rcnn_loss_cls.item()
        tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
        tb_dict['rcnn_loss'] = rcnn_loss.item()

        # tb_dict['reg_error_T1'] = reg_error_T1.item()
        # tb_dict['reg_error_T2'] = reg_error_T2.item()

        tb_dict['rcnn_loss_loc'] = loss_loc.item()
        tb_dict['rcnn_loss_angle'] = loss_angle.item()
        tb_dict['rcnn_loss_size'] = loss_size.item()
        tb_dict['rcnn_loss_corner'] = corner_loss.item()
        tb_dict['rcnn_loss_giou'] = gious_loss.item()

        tb_dict['rcnn_cls_fg'] = (cls_label > 0).sum().item()
        tb_dict['rcnn_cls_bg'] = (cls_label == 0).sum().item()

        visual_dict['rcnn_cls'] = rcnn_cls
        visual_dict['rcnn_reg'] = rcnn_reg
        visual_dict['pred_boxes3d'] = ret_dict['pred_boxes3d'].clone().view(-1, 7)

        return rcnn_loss

    def get_ioun_loss(model, ret_dict, tb_dict, visual_dict, input_data):

        rcnn_iou = ret_dict['rcnn_iou'].clone()
        rcnn_ref = ret_dict['rcnn_ref'].clone()
        gt_boxes3d = ret_dict['gt_boxes'].clone().view(-1, 7)
        pred_boxes3d = ret_dict['pred_boxes3d'].clone().view(-1, 7)
        refined_boxes3d = ret_dict['refined_box'].clone().view(-1, 7)
        reg_valid_mask = (ret_dict['cls'].float()).view(-1)
        iou_loss_dict = {}

        # iou mask
        batch_size = rcnn_iou.shape[0]
        fg_mask = reg_valid_mask > 0
        rcnn_iou = rcnn_iou
        rcnn_ref = rcnn_ref[fg_mask]
        gt_boxes3d = gt_boxes3d[fg_mask]
        pred_boxes3d = pred_boxes3d[fg_mask]
        refined_boxes3d = refined_boxes3d[fg_mask]
        fg_sum = torch.sum(fg_mask)
        if fg_sum != 0:

            # iou loss calculate

            # input box iou
            # iou2d, iou3d = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d, gt_boxes3d)
            # ref box iou
            iou2d, iou3d = iou3d_utils.boxes_iou3d_gpu(refined_boxes3d, gt_boxes3d)

            eye = torch.from_numpy(np.arange(0, iou3d.shape[0]).reshape(-1, 1)).long().cuda()
            iou3d = torch.gather(iou3d, 1, eye).detach()
            iou3d_label = iou3d
            iou3d_label = iou3d_label.pow(2)

            # origin
            # loss_iou = F.mse_loss(rcnn_iou[fg_mask].view(-1), iou3d_label.view(-1))*100

            # basic box refine
            loc_pred = pred_boxes3d[:, :3]
            siz_pred = pred_boxes3d[:, 3:6]
            ang_pred = pred_boxes3d[:, 6]
            loc_label = gt_boxes3d[:, :3]
            siz_label = gt_boxes3d[:, 3:6]
            ang_label = gt_boxes3d[:, 6]

            # loc
            loss_loc = F.smooth_l1_loss(rcnn_ref[:, :3], (loc_label - loc_pred) / siz_pred) * 300

            # size
            size_res_norm_label = (siz_label - siz_pred) / siz_pred
            size_res_norm = rcnn_ref[:, 3:6]
            loss_siz = F.smooth_l1_loss(size_res_norm, size_res_norm_label) * 300

            # ang
            angle_residual = ((ang_label) % np.pi - (ang_pred) % np.pi)
            loss_ang = F.smooth_l1_loss(rcnn_ref[:, 6], angle_residual) * 20

        else:
            loss_iou = torch.zeros(()).cuda()
            loss_loc = torch.zeros(()).cuda()
            loss_siz = torch.zeros(()).cuda()
            loss_ang = torch.zeros(()).cuda()

        loss_reg = loss_loc + loss_siz + loss_ang

        # BCE iou loss,BCE global iouloss,MSE iouloss,MSE global iouloss
        if True:
            gt_boxes3d = ret_dict['gt_boxes'].clone().view(-1, 7)
            invalid_mask = gt_boxes3d.sum(-1) != 0
            pred_boxes3d = ret_dict['pred_boxes3d'].clone().view(-1, 7)
            refined_boxes3d = ret_dict['refined_box'].clone().view(-1, 7)
            iou2d, iou3d = iou3d_utils.boxes_iou3d_gpu(refined_boxes3d, gt_boxes3d)
            eye = torch.from_numpy(np.arange(0, iou3d.shape[0]).reshape(-1, 1)).long().cuda()
            iou3d = torch.gather(iou3d, 1, eye).detach()
            iou3d_label = iou3d
            iou3d_label = iou3d_label.pow(2)
            # BCE
            # rcnn_cls_flat = rcnn_cls.view(-1)
            # batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), cls_label, reduction='none')
            # cls_valid_mask = (cls_label_flat >= 0).float()
            # rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

            # #BCEIOU
            # rcnn_cls_flat = reg_valid_mask.view(-1).clone()
            # if fg_sum != 0:
            #     rcnn_cls_flat[fg_mask]=iou3d_label.view(-1)
            # batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_iou), rcnn_cls_flat, reduction='none')
            # cls_valid_mask = (rcnn_cls_flat >= 0).float()
            # loss_iou = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

            # GlobalMSE
            # rcnn_cls_flat=iou3d_label.view(-1)
            # cls_valid_mask = ((fg_mask >= 0) | (~invalid_mask))
            # loss_iou = F.mse_loss(rcnn_iou[cls_valid_mask].view(-1), rcnn_cls_flat[cls_valid_mask].view(-1))*100

            # range MSE
            rcnn_cls_flat = iou3d_label.view(-1)
            cls_valid_mask = invalid_mask
            loss_iou = F.mse_loss(rcnn_iou[cls_valid_mask].view(-1), rcnn_cls_flat[cls_valid_mask].view(-1)) * 100

            # # #MSE
            # if fg_sum != 0:
            #     loss_iou = F.mse_loss(rcnn_iou[fg_mask].view(-1), iou3d_label.view(-1))*100
            # else:
            #     loss_iou = torch.zeros(()).cuda()
        else:
            raise NotImplementedError

        iou_loss_dict['ioun_loss_loc'] = loss_loc.item()
        iou_loss_dict['ioun_loss_siz'] = loss_siz.item()
        iou_loss_dict['ioun_loss_ang'] = loss_ang.item()
        iou_loss_dict['loss_iou'] = loss_iou.item()
        iou_loss_dict['loss_reg'] = loss_reg.item()

        tb_dict.update(iou_loss_dict)

        rcnn_loss_iou = loss_iou + loss_reg

        tb_dict['rcnn_loss_iou'] = rcnn_loss_iou.item()
        visual_dict['rcnn_iou'] = ret_dict['rcnn_iou'].clone()
        visual_dict['rcnn_ref'] = ret_dict['rcnn_ref'].clone()
        visual_dict['pred_boxes3d'] = ret_dict['pred_boxes3d'].clone().view(-1, 7)
        visual_dict['refined_box'] = ret_dict['refined_box'].clone().view(-1, 7)
        return rcnn_loss_iou

    return model_fn
