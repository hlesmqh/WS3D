import math
import numpy as np
import sys
import random
import torch

from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
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
        loss = torch.mean(1 - gious)
        #loss = F.smooth_l1_loss(1 - gious, torch.zeros_like(gious))
        return loss

class ious_3d_loss(nn.Module):
    # Compute the inter area of two rotated rectangles
    def _init_(self, gboxes, qboxes):
        super(ious_3d_loss, self)._init_()
        self.gboxes = gboxes
        self.qboxes = qboxes
        return

    def forward(ctx, gboxes, qboxes):
        ious_3D_object = ious_3D()
        ious = ious_3D_object(gboxes, qboxes)
        loss = torch.mean(1 - ious)
        # loss = F.smooth_l1_loss(1 - ious, torch.zeros_like(ious))
        return loss
    # gious_loss = gious_3d_loss(gboxes, qboxes_varible)
    # gious_loss.backward()


