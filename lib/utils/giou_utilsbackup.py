import math
import numpy as np
import sys
import random
import torch

from torch.autograd import Function
import torch.nn as nn

#from compute_ious import compute_ious_whih_shapely
from scipy.spatial import ConvexHull
from lib.utils.gious import *
class rbbox_corners_aligned(nn.Module):


    def _init_(self, gboxes):
        super(rbbox_corners_aligned, self)._init_()
        self.corners_gboxes = gboxes
        return

    def forward(ctx, gboxes):
        # generate clockwise corners and rotate it clockwise
        eps = 0.0
        N = gboxes.shape[0]
        center_x = gboxes[:, 0]
        center_y = gboxes[:, 1]
        x_d = gboxes[:, 2]
        y_d = gboxes[:, 3]
        corners = torch.zeros([N, 2, 4], device= gboxes.device, dtype=torch.float32)
        corners[:, 0, 0] = x_d.mul(-0.5)
        corners[:, 1, 0] = y_d.mul(-0.5)

        corners[:, 0, 1] = x_d.mul(-0.5)
        corners[:, 1, 1] = y_d.mul(0.5)

        corners[:, 0, 2] = x_d.mul(0.5)
        corners[:, 1, 2] = y_d.mul(0.5)

        corners[:, 0, 3] = x_d.mul(0.5)
        corners[:, 1, 3] = y_d.mul(-0.5)

        b = center_x.unsqueeze(1).repeat(1, 4).unsqueeze(1)
        c = center_y.unsqueeze(1).repeat(1, 4).unsqueeze(1)

        return (corners + torch.cat((b, c), 1))

## Transform the (cx, cy, w, l, theta) representation to 4 corners representation
class rbbox_to_corners(nn.Module):

    def _init_(self, rbbox):
        super(rbbox_to_corners, self)._init_()
        self.rbbox = rbbox
        return

    def forward(ctx, rbbox):

        assert rbbox.shape[1] == 5
        device  = rbbox.device
        corners = torch.zeros((rbbox.shape[0], 8), dtype=torch.float32, device = device)
        #with torch.no_grad():
        dxcos   = rbbox[:, 2].mul(torch.cos(rbbox[:, 4])) / 2.0
        dxsin   = rbbox[:, 2].mul(torch.sin(rbbox[:, 4])) / 2.0
        dycos   = rbbox[:, 3].mul(torch.cos(rbbox[:, 4])) / 2.0
        dysin   = rbbox[:, 3].mul(torch.sin(rbbox[:, 4])) / 2.0
        corners[:, 0] = -dxcos - dysin + rbbox[:, 0]
        corners[:, 1] =  dxsin - dycos + rbbox[:, 1]
        corners[:, 2] = -dxcos + dysin + rbbox[:, 0]
        corners[:, 3] =  dxsin + dycos + rbbox[:, 1]

        corners[:, 4] =  dxcos + dysin + rbbox[:, 0]
        corners[:, 5] = -dxsin + dycos + rbbox[:, 1]
        corners[:, 6] =  dxcos - dysin + rbbox[:, 0]
        corners[:, 7] = -dxsin - dycos + rbbox[:, 1]
            # generate clockwise corners and rotate it clockwise
        #ctx.save_for_backward(rbbox)
        return corners

class align_inter_aligned(nn.Module):

    def _init_(self, gboxes, qboxes):
        super(align_inter_aligned, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        return

    def forward(ctx, gboxes, qboxes):
        N = gboxes.shape[0]
        M = qboxes.shape[0]
        eps = 0.0000000000000001
        assert N == M

        ## we can project the 3D bounding boxes into 3 different plane
        ## view1 xoz plane
        inter_area_xoz = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        mbr_area_xoz = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        rbbox_corners_aligned_object = rbbox_corners_aligned()
        rotated_corners1 = rbbox_corners_aligned_object(gboxes[:, [0, 2, 3, 5, 6]])
        rotated_corners2 = rbbox_corners_aligned_object(qboxes[:, [0, 2, 3, 5, 6]])
        for i in range(N):
            iw = (min(rotated_corners1[i, 0, 1], rotated_corners2[i, 0, 3]) -
                  max(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 3]) + eps)
            if (iw > 0):
                ih = ((min(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                       max(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))
                if (ih > 0):
                    inter_area_xoz[i] = iw * ih

            iwmbr = (max(rotated_corners1[i, 0, 3], rotated_corners2[i, 0, 3]) -
                     min(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 0]) + eps)

            ihmbr = ((max(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                      min(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))
            mbr_area_xoz[i] = iwmbr * ihmbr

        ### view2 xoy plane
        inter_area_xoy = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        mbr_area_xoy = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        rotated_corners1 = rbbox_corners_aligned_object(gboxes[:, [0, 1, 3, 4, 6]])
        rotated_corners2 = rbbox_corners_aligned_object(qboxes[:, [0, 1, 3, 4, 6]])
        for i in range(N):
            iw = (min(rotated_corners1[i, 0, 1], rotated_corners2[i, 0, 3]) -
                  max(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 3]) + eps)
            if (iw > 0):
                ih = ((min(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                       max(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))
                if (ih > 0):
                    inter_area_xoy[i] = iw * ih

            iwmbr = (max(rotated_corners1[i, 0, 3], rotated_corners2[i, 0, 3]) -
                     min(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 0]) + eps)

            ihmbr = ((max(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                      min(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))

            mbr_area_xoy[i] = iwmbr * ihmbr

        ### view3 yoz plane
        inter_area_yoz = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        mbr_area_yoz = torch.zeros((N,), device=gboxes.device, dtype=torch.float32)
        rotated_corners1 = rbbox_corners_aligned_object(gboxes[:, [1, 2, 4, 5, 6]])
        rotated_corners2 = rbbox_corners_aligned_object(qboxes[:, [1, 2, 4, 5, 6]])
        for i in range(N):
            iw = (min(rotated_corners1[i, 0, 1], rotated_corners2[i, 0, 3]) -
                  max(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 3]) + eps)
            if (iw > 0):
                ih = ((min(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                       max(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))
                if (ih > 0):
                    inter_area_yoz[i] = iw * ih

            iwmbr = (max(rotated_corners1[i, 0, 3], rotated_corners2[i, 0, 3]) -
                     min(rotated_corners1[i, 0, 0], rotated_corners2[i, 0, 0]) + eps)

            ihmbr = ((max(rotated_corners1[i, 1, 1], rotated_corners2[i, 1, 1]) -
                      min(rotated_corners1[i, 1, 0], rotated_corners2[i, 1, 0]) + eps))
            mbr_area_yoz[i] = iwmbr * ihmbr

        return inter_area_xoz, mbr_area_xoz, inter_area_xoy, mbr_area_xoy, inter_area_yoz, mbr_area_yoz


class gious_3D(nn.Module):
    # Compute the inter area of two rotated rectangles
    def _init_(self, gboxes, qboxes, aligned=False):
        super(gious_3D, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        self.aligned = aligned
        return

    def forward(ctx, gboxes, qboxes, aligned=False):
        assert gboxes.shape[0] == qboxes.shape[0]
        indicator = torch.gt(gboxes[:, 3], 0) & torch.gt(gboxes[:, 4], 0) & torch.gt(gboxes[:, 5], 0) \
                    & torch.gt(qboxes[:, 3], 0) & torch.gt(qboxes[:, 4], 0) & torch.gt(qboxes[:, 5], 0)
        index_loc = torch.nonzero(indicator)
        ## if we want to compute the gious of two aligned rectangles
        gious = torch.zeros([gboxes.shape[0], ], device=gboxes.device, dtype=torch.float32)

        if (aligned):
            align_inter_aligned_object = align_inter_aligned()
            inter_area_xoz, mbr_area_xoz, inter_area_xoy, mbr_area_xoy, inter_area_yoz, mbr_area_yoz = align_inter_aligned_object(
                gboxes, qboxes)
            volume_gboxes = gboxes[:, 3].mul(gboxes[:, 4]).mul(gboxes[:, 5])
            volume_qboxes = qboxes[:, 3].mul(qboxes[:, 4]).mul(qboxes[:, 5])
            ## for three different views xoz plane
            # inter_h = (torch.min(gboxes[:, 1], qboxes[:, 1]) - torch.max(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4]))
            # oniou_h = (torch.max(gboxes[:, 1], qboxes[:, 1]) - torch.min(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4]))
            # inter_h[inter_h < 0] = 0
            # oniou_h[oniou_h < 0] = 0
            # inter_area_xoz_cuda = inter_area_xoz.to(torch.device(gboxes.device))
            # mbr_area_xoz_cuda   = mbr_area_xoz.to(torch.device(gboxes.device))
            # volume_inc = inter_h.mul(inter_area_xoz_cuda)
            # volume_con = oniou_h.mul(mbr_area_xoz_cuda)
            # volume_union = (volume_gboxes + volume_qboxes - volume_inc)
            # volume_ca = volume_con - volume_union
            # ious = torch.div(volume_inc, volume_union)
            union_xoz = gboxes[:, 3].mul(gboxes[:, 5]) + qboxes[:, 3].mul(qboxes[:, 5]) - inter_area_xoz
            iou_xoz = torch.div(inter_area_xoz, union_xoz)
            iou_bis_xoz = torch.div(mbr_area_xoz - union_xoz, mbr_area_xoz)
            gious_xoz = iou_xoz - iou_bis_xoz
            ## for xoy plane
            union_xoy = gboxes[:, 3].mul(gboxes[:, 4]) + qboxes[:, 3].mul(qboxes[:, 4]) - inter_area_xoy
            iou_xoy = torch.div(inter_area_xoy, union_xoy)
            iou_bis_xoy = torch.div(mbr_area_xoy - union_xoy, mbr_area_xoy)
            gious_xoy = iou_xoy - iou_bis_xoy
            ## for yoz plane
            union_yoz = gboxes[:, 4].mul(gboxes[:, 5]) + qboxes[:, 4].mul(qboxes[:, 5]) - inter_area_xoy
            iou_yoz = torch.div(inter_area_yoz, union_yoz)
            iou_bis_yoz = torch.div(mbr_area_yoz - union_yoz, mbr_area_yoz)
            gious_xoy = iou_yoz - iou_bis_yoz
            gious[index_loc[:, 0]] = (gious_xoz[index_loc[:, 0]] + gious_xoy[index_loc[:, 0]] + gious_xoy[
                index_loc[:, 0]]) / 3.0

            # for i in range(inter_area_xoz.shape[0]):
            #    if (gious[i] > 1):
            #        print("infor: (%.4f %.4f %.4f %.4f %.4f %.4f %.4f,%.4f %.4f %.4f %.4f)"
            #              % (i, inter_h[i], oniou_h[i], inter_area_xoz[i], mbr_area_xoz[i], ious[i], gious[i], volume_inc[i],
            #                 volume_con[i], volume_union[i], volume_ca[i]))
            #    elif (gious[i] < -1):
            #        print("infor: (%.4f %.4f %.4f %.4f %.4f %.4f %.4f,%.4f %.4f %.4f %.4f)"
            #              % (i, inter_h[i], oniou_h[i], inter_area_xoz[i], mbr_area_xoz[i], ious[i], gious[i], volume_inc[i],
            #                 volume_con[i], volume_union[i], volume_ca[i]))
        else:
            rbbox_to_corners_object = rbbox_to_corners()
            corners_gboxes = rbbox_to_corners_object(gboxes[:, [0, 2, 3, 5, 6]])
            corners_qboxes = rbbox_to_corners_object(qboxes[:, [0, 2, 3, 5, 6]])
            # compute the inter area
            rinter_area_compute_object = rinter_area_compute()
            inter_area = rinter_area_compute_object(corners_gboxes, corners_qboxes)

            corners_gboxes_1 = torch.stack((corners_gboxes[:, [0, 2, 4, 6]], corners_gboxes[:, [1, 3, 5, 7]]), 2)
            corners_qboxes_1 = torch.stack((corners_qboxes[:, [0, 2, 4, 6]], corners_qboxes[:, [1, 3, 5, 7]]), 2)
            corners_pts = torch.cat((corners_gboxes_1, corners_qboxes_1), 1)

            # compute the mbr area
            mbr_area_compute_object = mbr_area_compute()
            mbr_area = mbr_area_compute_object(corners_pts)

            ## Compute the gious for 3D
            inter_h = (torch.min(gboxes[:, 1], qboxes[:, 1]) - torch.max(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4]))
            oniou_h = (torch.max(gboxes[:, 1], qboxes[:, 1]) - torch.min(gboxes[:, 1] - gboxes[:, 4], qboxes[:, 1] - qboxes[:, 4]))
            inter_h[inter_h < 0] = 0
            volume_gboxes = gboxes[:, 3].mul(gboxes[:, 4]).mul(gboxes[:, 5])
            volume_qboxes = qboxes[:, 3].mul(qboxes[:, 4]).mul(qboxes[:, 5])
            inter_area_cuda = inter_area.to(torch.device(gboxes.device))
            mbr_area_cuda   = mbr_area.to(torch.device(gboxes.device))
            volume_inc = inter_h.mul(inter_area_cuda)
            volume_con = oniou_h.mul(mbr_area_cuda)
            volume_union = (volume_gboxes + volume_qboxes - volume_inc)
            volume_ca    = volume_con - volume_union
            ious         = torch.div(volume_inc, volume_union)

            gious = torch.zeros([gboxes.shape[0],], device=gboxes.device, dtype=torch.float32)
            gious[index_loc[:, 0]] = ious[index_loc[:, 0]] - torch.div(volume_ca[index_loc[:, 0]], volume_con[index_loc[:, 0]])
            # for i in range(inter_area.shape[0]):
            #     if(gious[i] < -1):
            #         print("infor: (%.4f %.4f %.4f %.4f %.4f %.4f %.4f,%.4f %.4f %.4f %.4f)"
            #           %(i,inter_h[i], oniou_h[i], inter_area[i], mbr_area[i], ious[i], gious[i],volume_inc[i],volume_con[i],volume_union[i],volume_ca[i]))

        return torch.unsqueeze(gious, 1)


class gious_3d_loss(nn.Module):
    # Compute the inter area of two rotated rectangles
    def _init_(self, gboxes, qboxes):
        super(gious_3d_loss, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        return

    def forward(ctx, gboxes, qboxes):
        gious_3D_object = gious_3D()
        gious = gious_3D_object(gboxes, qboxes)
        loss = torch.mean(gious)
        return loss

    # gious_loss = gious_3d_loss(gboxes, qboxes_varible)
    # gious_loss.backward()


class Combine_WeightedL1And_GiousLoss(Loss):
    """
     combine the L1 loss and Gious Loss
     Define the GIOU loss  for object detection. Details can be found as below:
     Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression, CVPR 2019
     Which define a general 2D bounding box IOU as the loss function for 2D objects detection.
     Compute loss function.
      Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors, code_size]
        representing the (encoded) predicted locations of objects.
        target_tensor: A float tensor of shape [batch_size, num_anchors, code_size]
        representing the regression targets
        batch_anchors: A float tensor of shape [batch_size, num_anchors, code_size]
        representing the standard anchors bounding boxes
        labels: A float tensor of shape [batch_size, num_anchors] representing
        the class label of standard anchors
        weights: a float tensor of shape [batch_size, num_anchors]
      Returns:
        loss: a float tensor of shape [batch_size, num_anchors] tensor
          representing the value of the loss function.
      """

    def __init__(self, sigma=3.0, code_weights=None, codewise=True):
        super().__init__()
        self._sigma = sigma
        if code_weights is not None:
            self._code_weights = np.array(code_weights, dtype=np.float32)
            self._code_weights = Variable(torch.from_numpy(self._code_weights).cuda())
        else:
            self._code_weights = None
        self._codewise = codewise

    def _compute_loss(self, prediction_tensor, target_tensor, batch_anchors, labels, rects, Trv2cs, weights=None):
        # box decoder

        num_samples_batchs = batch_anchors.shape[0]
        num_anchors = batch_anchors.shape[1]
        # if torch.cuda.is_available():
        device = prediction_tensor.device
        ious_tensor = torch.zeros([num_samples_batchs, num_anchors, 1], device=device, dtype=torch.float32)
        aligned = False
        IOU_LOSS = True
        for sample_i in range(0, num_samples_batchs):
            pos_index = labels[sample_i, :] > 0
            index_loc = torch.nonzero(pos_index)
            if (index_loc.shape[0] < 1):
                continue

            prediction_tensor_valid = prediction_tensor[sample_i, index_loc[:, 0], :]
            target_tensor_valid = target_tensor[sample_i, index_loc[:, 0], :]
            batch_anchors_valid = batch_anchors[sample_i, index_loc[:, 0], :]
            if weights is not None:
                weights_valid = weights[sample_i, index_loc[:, 0]].unsqueeze(-1)
            # prediction_tensor_valid_test = torch.zeros([prediction_tensor_valid.shape[0], prediction_tensor_valid.shape[1]], device=device, dtype=torch.float32)
            # prediction_3d_box_sample_i   = box_np_ops._second_box_decode(prediction_tensor_valid, batch_anchors_valid, False, True)
            # ground_truth_3d_box_sample_i = box_np_ops._second_box_decode(target_tensor_valid, batch_anchors_valid, False, True)
            _second_box_decode_operation_object = _second_box_decode_operation()
            prediction_3d_box_sample_i = _second_box_decode_operation_object(prediction_tensor_valid,
                                                                             batch_anchors_valid, False, True)
            ground_truth_3d_box_sample_i = _second_box_decode_operation_object(target_tensor_valid, batch_anchors_valid,
                                                                               False, True)
            # transform the boxes in lidar coordinate to camera coordinae
            prediction_3d_box_sample_i_camera = box_torch_ops.box_lidar_to_camera(prediction_3d_box_sample_i,
                                                                                  rects[sample_i], Trv2cs[sample_i])
            ground_truth_3d_box_sample_i_camera = box_torch_ops.box_lidar_to_camera(ground_truth_3d_box_sample_i,
                                                                                    rects[sample_i], Trv2cs[sample_i])

            # prediction_3d_box_sample_i_camera_   = prediction_3d_box_sample_i_camera[:, [0, 1, 2, 4, 5, 3, 6]]
            # ground_truth_3d_box_sample_i_camera_ = ground_truth_3d_box_sample_i_camera[:,[0, 1, 2, 4, 5, 3, 6]]

            # delta_angle = torch.FloatTensor([0, 0, 0, 0, 0, 0, -3.141592654 / 2.0])
            # delta_angle_tensor = delta_angle.repeat(prediction_3d_box_sample_i.shape[0], 1)
            # delta_angle_tensor_cuda = delta_angle_tensor.to(torch.device(device))
            # prediction_3d_box_sample_i_ = prediction_3d_box_sample_i + delta_angle_tensor_cuda
            # ground_truth_3d_box_sample_i_ = ground_truth_3d_box_sample_i + delta_angle_tensor_cuda
            # iou_sample_i = d3_box_overlap_simple_torch(prediction_3d_box_sample_i, ground_truth_3d_box_sample_i, -1)
            # iou_sample_i = d3_box_overlap_general_torch(prediction_3d_box_sample_i, ground_truth_3d_box_sample_i, -1)
            if (IOU_LOSS):
                ious_3D_object = ious_3D()
                iou_sample_i = ious_3D_object(ground_truth_3d_box_sample_i_camera, prediction_3d_box_sample_i_camera,
                                              aligned)
            else:
                gious_3D_object = gious_3D()
                iou_sample_i = gious_3D_object(ground_truth_3d_box_sample_i_camera, prediction_3d_box_sample_i_camera,
                                               aligned)

            # print("shape0 = ", iou_sample_i.shape[0])
            # print("iou_sample_i=", iou_sample_i)
            # iou_loss     = torch.ones([iou_sample_i.shape[0], 1], device=device, dtype=torch.float32)*2.71828182  - torch.exp(iou_sample_i)
            iou_loss = torch.ones([iou_sample_i.shape[0], 1], device=device, dtype=torch.float32) - iou_sample_i
            # print("shape0 = ", iou_loss.shape)
            # print("iou_loss = ", torch.transpose(iou_loss, 1, 0) )
            # iou_loss[iou_sample_i < 0] = 0
            # iou_loss[iou_sample_i > 2] = 2
            if weights is not None:
                iou_loss = iou_loss.mul(weights_valid) * 1.0 / 7.0
            ious_tensor[sample_i, index_loc[:, 0], 0] = iou_loss.squeeze()
            # first_ind = index_loc.clone()
            # for ind in range(index_loc.shape[0]):
            #    first_ind[ind, 0] = sample_i
            # tuple_index = (first_ind[:, 0], index_loc[:, 0])
            # values = iou_loss[:, 0]
            # ious_tensor[:, :, 0].index_put_(tuple_index, values)

        ious_tensor = ious_tensor.repeat(1, 1, 7)
        '''
        rad_pred_encoding = torch.sin(prediction_tensor[..., -1:]) * torch.cos(target_tensor[..., -1:])
        rad_tg_encoding   = torch.cos(prediction_tensor[..., -1:]) * torch.sin(target_tensor[..., -1:])
        prediction_tensor_1 = torch.cat([prediction_tensor[..., :-1], rad_pred_encoding], dim=-1)
        target_tensor_1     = torch.cat([target_tensor[..., :-1], rad_tg_encoding], dim=-1)

         #sin(a - b) = sinacosb-cosasinb
        #prediction_tensor, target_tensor = add_sin_difference(prediction_tensor, target_tensor)
            ## L1 loss computation
        diff = prediction_tensor_1 - target_tensor_1
        if self._code_weights is not None:
          code_weights = self._code_weights.type_as(prediction_tensor)
          diff = code_weights.view(1, 1, -1) * diff
        abs_diff = torch.abs(diff)
        abs_diff_lt_1 = torch.le(abs_diff, 1 / (self._sigma**2)).type_as(abs_diff)
        loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) \
          + (abs_diff - 0.5 / (self._sigma**2)) * (1. - abs_diff_lt_1)
        if self._codewise:
          anchorwise_smooth_l1norm = loss
          if weights is not None:
            anchorwise_smooth_l1norm *= weights.unsqueeze(-1)
        else:
          anchorwise_smooth_l1norm = torch.sum(loss, 2)#  * weights
          if weights is not None:
            anchorwise_smooth_l1norm *= weights

        num_samples_batchs = batch_anchors.shape[0]
        num_anchors = batch_anchors.shape[1]

        alpha = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0])
        beta  = 1.0
        alpha_tensor = alpha.repeat(num_samples_batchs, num_anchors, 1)

        alpha_tensor_cuda = alpha_tensor.to(torch.device(device))
        #loc_loss_l1_reduced   = anchorwise_smooth_l1norm.sum() / num_samples_batchs
        #locloss_gious_reduced = ious_tensor.sum() / num_samples_batchs
        #print("loc_loss_l1_reduced = %.4f locloss_gious_reduced = %.4f" %(loc_loss_l1_reduced, locloss_gious_reduced))
        #anchorwise_smooth_l1norm[:, :, 0:6] = 0.0
        #loc_loss_l1_reduced   = anchorwise_smooth_l1norm.sum() / num_samples_batchs
        #locloss_gious_reduced = ious_tensor.sum() / num_samples_batchs
        #print("locloss_gious_reduced = %.4f" %(locloss_gious_reduced))

        '''
        return ious_tensor  # + alpha_tensor_cuda.mul(anchorwise_smooth_l1norm)