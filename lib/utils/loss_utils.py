import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import lib.utils.kitti_utils as kitti_utils

class DiceLoss(nn.Module):
    def __init__(self, ignore_target=-1):
        super().__init__()
        self.ignore_target = ignore_target

    def forward(self, input, target):
        """
        :param input: (N), logit
        :param target: (N), {0, 1}
        :return:
        """
        input = torch.sigmoid(input.view(-1))
        target = target.float().view(-1)
        mask = (target != self.ignore_target).float()
        return 1.0 - (torch.min(input, target) * mask).sum() / torch.clamp((torch.max(input, target) * mask).sum(), min=1.0)


class SigmoidFocalClassificationLoss(nn.Module):
    """Sigmoid focal cross entropy loss.
      Focal loss down-weights well classified examples and focusses on the hard
      examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.
        Args:
            gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
            alpha: optional alpha weighting factor to balance positives vs negatives.
            all_zero_negative: bool. if True, will treat all zero as background.
            else, will treat first label as background. only affect alpha.
        """
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self,
                prediction_tensor,
                target_tensor,
                weights):
        """Compute loss function.

        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing the predicted logits for each class
            target_tensor: A float tensor of shape [batch_size, num_anchors,
              num_classes] representing one-hot encoded classification targets
            weights: a float tensor of shape [batch_size, num_anchors]
            class_indices: (Optional) A 1-D integer tensor of class indices.
              If provided, computes loss only for the specified class indices.

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor))
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) +
               ((1 - target_tensor) * (1 - prediction_probabilities)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha))

        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)
        return focal_cross_entropy_loss * weights


def _sigmoid_cross_entropy_with_logits(logits, labels):
    # to be compatible with tensorflow, we don't use ignore_idx
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    # transpose_param = [0] + [param[-1]] + param[1:-1]
    # logits = logits.permute(*transpose_param)
    # loss_ftor = nn.NLLLoss(reduce=False)
    # loss = loss_ftor(F.logsigmoid(logits), labels)
    return loss


def get_rpn_reg_loss(pred_reg, reg_label, loc_scope, loc_bin_size):
    """
    Bin-based 3D bounding boxes regression loss. See https://arxiv.org/abs/1812.04244 for more details.

    :param pred_reg: (N, C)
    :param reg_label: (N, 3) [dx, 0, dz]
    :param loc_scope: constant
    :param loc_bin_size: constant
    :param get_xz_fine:
    :return:
    """
    per_loc_bin_num = int((loc_scope + 1e-3) / loc_bin_size) * 2

    reg_loss_dict = {}
    loc_loss = 0

    # xz label loading
    x_offset_label, y_offset_label, z_offset_label = reg_label[:, 0], reg_label[:, 1], reg_label[:, 2]

    # rpn regression parsing
    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
    z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
    start_offset = z_res_r

    # rpn regression output translation bin
    x_shift = torch.clamp(x_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
    z_shift = torch.clamp(z_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
    x_bin_label = (x_shift / loc_bin_size).floor().long()
    z_bin_label = (z_shift / loc_bin_size).floor().long()

    # rpn bin loss
    loss_x_bin = F.cross_entropy(pred_reg[:, x_bin_l: x_bin_r], x_bin_label)
    loss_z_bin = F.cross_entropy(pred_reg[:, z_bin_l: z_bin_r], z_bin_label)
    reg_loss_dict['loss_x_bin'] = loss_x_bin.item()
    reg_loss_dict['loss_z_bin'] = loss_z_bin.item()
    loc_loss += loss_x_bin + loss_z_bin

    # rpn regression output translation residual
    x_res_label = x_shift - (x_bin_label.float() * loc_bin_size + loc_bin_size / 2)
    z_res_label = z_shift - (z_bin_label.float() * loc_bin_size + loc_bin_size / 2)
    x_res_norm_label = x_res_label / (loc_bin_size / 2)
    z_res_norm_label = z_res_label / (loc_bin_size / 2)

    # rpn residual loss
    x_bin_onehot = torch.cuda.FloatTensor(x_bin_label.size(0), per_loc_bin_num).zero_()
    x_bin_onehot.scatter_(1, x_bin_label.view(-1, 1).long(), 1)
    z_bin_onehot = torch.cuda.FloatTensor(z_bin_label.size(0), per_loc_bin_num).zero_()
    z_bin_onehot.scatter_(1, z_bin_label.view(-1, 1).long(), 1)

    loss_x_res = F.smooth_l1_loss((pred_reg[:, x_res_l: x_res_r] * x_bin_onehot).sum(dim=1), x_res_norm_label)
    loss_z_res = F.smooth_l1_loss((pred_reg[:, z_res_l: z_res_r] * z_bin_onehot).sum(dim=1), z_res_norm_label)
    reg_loss_dict['loss_x_res'] = loss_x_res.item()
    reg_loss_dict['loss_z_res'] = loss_z_res.item()
    loc_loss += loss_x_res + loss_z_res

    #rpn output assert
    assert pred_reg.shape[1] == start_offset, '%d vs %d' % (pred_reg.shape[1], start_offset)

    return loc_loss, reg_loss_dict


def get_rcnn_reg_loss(pred_reg, reg_label, loc_scope, loc_bin_size, num_head_bin, anchor_size,
                 get_xz_fine=True, get_y_by_bin=False, loc_y_scope=0.5, loc_y_bin_size=0.25, get_ry_fine=False):

    """
    Bin-based 3D bounding boxes regression loss. See https://arxiv.org/abs/1812.04244 for more details.
    
    :param pred_reg: (N, C)
    :param reg_label: (N, 7) [dx, dy, dz, h, w, l, ry]
    :param loc_scope: constant
    :param loc_bin_size: constant
    :param num_head_bin: constant
    :param anchor_size: (N, 3) or (3)
    :param get_xz_fine:
    :param get_y_by_bin:
    :param loc_y_scope:
    :param loc_y_bin_size:
    :param get_ry_fine:
    :return:
    """
    per_loc_bin_num = int((loc_scope + 1e-3) / loc_bin_size) * 2
    loc_y_bin_num = int((loc_y_scope + 1e-3) / loc_y_bin_size) * 2

    reg_loss_dict = {}
    loc_loss = 0

    # xz localization loss
    x_offset_label, y_offset_label, z_offset_label = reg_label[:, 0], reg_label[:, 1], reg_label[:, 2]

    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2

    x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
    z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
    start_offset = z_res_r
    if get_xz_fine:
        x_shift = torch.clamp(x_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
        z_shift = torch.clamp(z_offset_label + loc_scope, 0, loc_scope * 2 - 1e-3)
        x_bin_label = (x_shift / loc_bin_size).floor().long()
        z_bin_label = (z_shift / loc_bin_size).floor().long()

        loss_x_bin = F.cross_entropy(pred_reg[:, x_bin_l: x_bin_r], x_bin_label)
        loss_z_bin = F.cross_entropy(pred_reg[:, z_bin_l: z_bin_r], z_bin_label)
        reg_loss_dict['loss_x_bin'] = loss_x_bin.item()
        reg_loss_dict['loss_z_bin'] = loss_z_bin.item()
        loc_loss += loss_x_bin + loss_z_bin

        x_res_label = x_shift - (x_bin_label.float() * loc_bin_size + loc_bin_size / 2)
        z_res_label = z_shift - (z_bin_label.float() * loc_bin_size + loc_bin_size / 2)
        x_res_norm_label = x_res_label / (loc_bin_size / 2)
        z_res_norm_label = z_res_label / (loc_bin_size / 2)

        x_bin_onehot = torch.cuda.FloatTensor(x_bin_label.size(0), per_loc_bin_num).zero_()
        x_bin_onehot.scatter_(1, x_bin_label.view(-1, 1).long(), 1)
        z_bin_onehot = torch.cuda.FloatTensor(z_bin_label.size(0), per_loc_bin_num).zero_()
        z_bin_onehot.scatter_(1, z_bin_label.view(-1, 1).long(), 1)

        loss_x_res = F.smooth_l1_loss((pred_reg[:, x_res_l: x_res_r] * x_bin_onehot).sum(dim=1), x_res_norm_label)
        loss_z_res = F.smooth_l1_loss((pred_reg[:, z_res_l: z_res_r] * z_bin_onehot).sum(dim=1), z_res_norm_label)
        reg_loss_dict['loss_x_res'] = loss_x_res.item()
        reg_loss_dict['loss_z_res'] = loss_z_res.item()
        loc_loss += loss_x_res + loss_z_res

    else:
        TEST_SMOOTH_XZ = True
        if TEST_SMOOTH_XZ:
            loc_loss=0
            x_offset_label = x_offset_label / loc_scope
            z_offset_label = z_offset_label / loc_scope
            loss_x_offset = F.smooth_l1_loss(pred_reg[:, x_res_l: x_res_l+1].sum(dim=1), x_offset_label)
            reg_loss_dict['loss_x_offset'] = loss_x_offset.item()
            loc_loss += loss_x_offset
            loss_z_offset = F.smooth_l1_loss(pred_reg[:, z_res_l: z_res_l+1].sum(dim=1), z_offset_label)
            reg_loss_dict['loss_z_offset'] = loss_z_offset.item()
            loc_loss += loss_z_offset


    # y localization loss
    if get_y_by_bin:
        y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
        y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
        start_offset = y_res_r

        y_shift = torch.clamp(y_offset_label + loc_y_scope, 0, loc_y_scope * 2 - 1e-3)
        y_bin_label = (y_shift / loc_y_bin_size).floor().long()
        y_res_label = y_shift - (y_bin_label.float() * loc_y_bin_size + loc_y_bin_size / 2)
        y_res_norm_label = y_res_label / loc_y_bin_size

        y_bin_onehot = torch.cuda.FloatTensor(y_bin_label.size(0), loc_y_bin_num).zero_()
        y_bin_onehot.scatter_(1, y_bin_label.view(-1, 1).long(), 1)

        loss_y_bin = F.cross_entropy(pred_reg[:, y_bin_l: y_bin_r], y_bin_label)
        loss_y_res = F.smooth_l1_loss((pred_reg[:, y_res_l: y_res_r] * y_bin_onehot).sum(dim=1), y_res_norm_label)

        reg_loss_dict['loss_y_bin'] = loss_y_bin.item()
        reg_loss_dict['loss_y_res'] = loss_y_res.item()

        loc_loss += loss_y_bin + loss_y_res
    else:


        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r

        loss_y_offset = F.mse_loss(pred_reg[:, y_offset_l: y_offset_r].sum(dim=1), y_offset_label)# - reg_label[:,3]/2)
        reg_loss_dict['loss_y_offset'] = loss_y_offset.item()
        loc_loss += loss_y_offset

    # angle loss
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

    ry_label = reg_label[:, 6]

    if get_ry_fine:
        # # divide pi into several bins
        angle_per_class = np.pi / num_head_bin

        ry_label = ry_label % np.pi  # (0 ~ pi)


        shift_angle = torch.clamp(ry_label, min=1e-3, max=np.pi - 1e-3)  # (0, pi/2)

        # bin center is (5, 10, 15, ..., 85)
        ry_bin_label = (shift_angle / angle_per_class).floor().long()
        ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

        # # divide pi/2 into several bins
        # angle_per_class = (np.pi / 2) / num_head_bin
        #
        # ry_label = ry_label % (2 * np.pi)  # 0 ~ 2pi
        # opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
        # ry_label[opposite_flag] = (ry_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        # shift_angle = (ry_label + np.pi * 0.5) % (2 * np.pi)  # (0 ~ pi)
        #
        # shift_angle = torch.clamp(shift_angle - np.pi * 0.25, min=1e-3, max=np.pi * 0.5 - 1e-3)  # (0, pi/2)
        #
        # # bin center is (5, 10, 15, ..., 85)
        # ry_bin_label = (shift_angle / angle_per_class).floor().long()
        # ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        # ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    else:
        # divide 2pi into several bins
        angle_per_class = (2 * np.pi) / num_head_bin
        heading_angle = ry_label % (2 * np.pi)  # 0 ~ 2pi

        shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
        ry_bin_label = (shift_angle / angle_per_class).floor().long()
        ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    ry_bin_onehot = torch.cuda.FloatTensor(ry_bin_label.size(0), num_head_bin).zero_()
    ry_bin_onehot.scatter_(1, ry_bin_label.long().view(-1, 1), 1)
    loss_ry_bin = F.cross_entropy(pred_reg[:, ry_bin_l:ry_bin_r], ry_bin_label)
    loss_ry_res = F.smooth_l1_loss((pred_reg[:, ry_res_l: ry_res_r] * ry_bin_onehot).sum(dim=1), ry_res_norm_label)

    reg_loss_dict['loss_ry_bin'] = loss_ry_bin.item()
    reg_loss_dict['loss_ry_res'] = loss_ry_res.item()
    angle_loss = loss_ry_bin + loss_ry_res

    # size loss
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3

    # iou loss position
    #assert pred_reg.shape[1] == (size_res_r + 1), '%d vs %d' % (pred_reg.shape[1], size_res_r + 1)

    # L1 norm size
    size_res_norm_label = (reg_label[:, 3:6] - anchor_size) / anchor_size
    size_res_norm = pred_reg[:, size_res_l:size_res_r]
    size_loss = F.smooth_l1_loss(size_res_norm[:, :3], size_res_norm_label[:, :3])

    # Log norm size
    # size_res_norm_label = torch.log(reg_label[:, 3:6] / anchor_size)
    # size_res_norm = pred_reg[:, size_res_l:size_res_r]
    # size_loss = F.smooth_l1_loss(size_res_norm[:, :3], size_res_norm_label[:, :3])

    # loss_w = F.mse_loss(size_res_norm[:, 0], size_res_norm_label[:, 0])
    # loss_h = F.mse_loss(size_res_norm[:, 1], size_res_norm_label[:, 1])
    # loss_l = F.mse_loss(size_res_norm[:, 2], size_res_norm_label[:, 2])
    # size_loss = loss_w + loss_h + loss_l
    #
    loc_loss = loc_loss
    angle_loss = angle_loss
    size_loss = size_loss


    return loc_loss, angle_loss, size_loss, reg_loss_dict



# def get_bbox_iou_loss(pred_reg, pred_ref, pred_iou, reg_label, loc_scope, loc_bin_size, num_head_bin, anchor_size,
#                        loc_y_scope=0.5, loc_y_bin_size=0.25):
# # def get_bbox_iou_loss(pred_reg, pred_iou, reg_label, loc_scope, loc_bin_size, num_head_bin, anchor_size,
# #                       loc_y_scope=0.5, loc_y_bin_size=0.25):
#     """
#     :param roi_box3d: (N, 7)
#     :param pred_reg: (N, C)
#     :param loc_scope:
#     :param loc_bin_size:
#     :param num_head_bin:
#     :param anchor_size:
#     :param get_xz_fine:
#     :param get_y_by_bin:
#     :param loc_y_scope:
#     :param loc_y_bin_size:
#     :param get_ry_fine:
#     :return:
#     """
#     iou_loss_dict = {}
#
#     anchor_size = anchor_size.to(pred_reg.get_device())
#     per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
#
#     x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
#     z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
#     start_offset = z_res_r
#
#     pos_x = pred_reg[:, x_res_l] * loc_scope
#     pos_z = pred_reg[:, z_res_l] * loc_scope
#
#     y_offset_l, y_offset_r = start_offset, start_offset + 1
#     start_offset = y_offset_r
#
#     pos_y =  pred_reg[:, y_offset_l]
#
#     # recover ry rotation
#     ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
#     ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin
#
#     ry_bin = torch.argmax(pred_reg[:, ry_bin_l: ry_bin_r], dim=1)
#     ry_res_norm = torch.gather(pred_reg[:, ry_res_l: ry_res_r], dim=1, index=ry_bin.unsqueeze(dim=1)).squeeze(dim=1)
#
#     angle_per_class = (2 * np.pi) / num_head_bin
#     ry_res = ry_res_norm * (angle_per_class / 2)
#
#     # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
#     ry = ((ry_bin.float() * angle_per_class + ry_res)) % (2 * np.pi)
#     ry[ry > np.pi] -= 2 * np.pi
#
#     # # recover size
#     size_res_l, size_res_r = ry_res_r, ry_res_r + 3
#
#     #recover iou
#     #assert (size_res_r+1) == pred_reg.shape[1]
#
#     size_res_norm = pred_reg[:, size_res_l: size_res_r]
#     hwl = size_res_norm * anchor_size + anchor_size
#
#     # # shift to original coords
#     shift_ret_box3d = torch.cat((pos_x.view(-1, 1), pos_y.view(-1, 1), pos_z.view(-1, 1), hwl, ry.view(-1, 1)), dim=1)
#     ret_box3d = shift_ret_box3d
#     ret_box3d[:,1] += ret_box3d[:,3]/2
#
#     iou2d, iou3d = iou3d_utils.boxes_iou3d_gpu(ret_box3d, reg_label.squeeze(1))
#     eye = torch.from_numpy(np.arange(0, iou3d.shape[0]).reshape(-1, 1)).long().cuda()
#
#     iou3d = torch.gather(iou3d, 1, eye).detach()
#     iou3d_label = iou3d
#
#     #square
#     iou3d_label = iou3d_label.pow(2)
#
#     # halfsoft label
#     #iou3d_label[iou3d_label < 0.7] = 0.0
#
#     # hard label
#     # iou3d_label = (iou3d_label > 0.7).float()
#
#     loss_iou_score = F.mse_loss(pred_iou.view(-1), iou3d_label.view(-1))
#     # loss_iou = F.smooth_l1_loss(pred_reg[:, -1].view(-1), iou3d.view(-1))
#     # loss_iou = F.smooth_l1_loss(pred_reg[:, -1], iou3d)
#     # loss_iou = F.binary_cross_entropy(torch.sigmoid(pred_reg[:, -1]).view(-1), iou3d.view(-1), reduction='none').sum()
#     iou_loss_dict['loss_iou'] = loss_iou_score.item()
#
#     # reg_label = reg_label.view(-1, 7)
#     # #y_ref = ((ret_box3d[:,1]+ret_box3d[:,3]/2)-(reg_label[:,1]+reg_label[:,3]/2))/(ret_box3d[:,3]/2)
#     # loss_iou_ref = F.l1_loss(pred_ref[:, 0], y_ref)
#     # #43
#     # #h_ref = ret_box3d[:,3]/reg_label[:,3]
#     # #44
#     # #h_ref = (ret_box3d[:, 3]-reg_label[:, 3]) / ret_box3d[:, 3]
#     # loss_iou_ref += F.l1_loss(pred_ref[:, 1], h_ref)
#
#     #only ref the IOU>0.5
#     roi_ry = ret_box3d[:, 6] % (2 * np.pi)
#     gt_boxes3d_ct = reg_label.clone()
#     gt_boxes3d_ct[:, 0:3] = gt_boxes3d_ct[:, 0:3] - ret_box3d[:, 0:3]
#     # rotate to the direction of head
#     gt_boxes3d_ct = kitti_utils.rotate_pc_along_y_torch(gt_boxes3d_ct.reshape(-1, 1, 7),
#                                                         roi_ry.reshape(-1)).reshape(-1, 7)
#     gt_boxes3d_ct[:, 6] = gt_boxes3d_ct[:, 6] - roi_ry
#     ioumask = (iou3d_label>0.25).view(-1)
#     if ioumask.long().sum()>0:
#         pred_ref = pred_ref[ioumask]
#         anchor_size = ret_box3d[ioumask][:, 0:3]
#         reg_label = gt_boxes3d_ct[ioumask]
#
#         per_loc_bin_num = int((loc_scope + 1e-3) / loc_bin_size) * 2
#         loc_y_bin_num = int((loc_y_scope + 1e-3) / loc_y_bin_size) * 2
#
#         reg_loss_dict = {}
#         loc_loss = 0
#
#         # xz localization loss
#         x_offset_label, y_offset_label, z_offset_label = reg_label[:, 0], reg_label[:, 1], reg_label[:, 2]
#
#         x_bin_l, x_bin_r = 0, per_loc_bin_num
#         z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
#
#         x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
#         z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
#         start_offset = z_res_r
#
#         loc_loss = 0
#         x_offset_label = x_offset_label / loc_scope
#         z_offset_label = z_offset_label / loc_scope
#         loss_x_offset = F.smooth_l1_loss(pred_ref[:, x_res_l: x_res_l + 1].sum(dim=1), x_offset_label)
#         reg_loss_dict['loss_x_offset'] = loss_x_offset.item()
#         loc_loss += loss_x_offset
#         loss_z_offset = F.smooth_l1_loss(pred_ref[:, z_res_l: z_res_l + 1].sum(dim=1), z_offset_label)
#         reg_loss_dict['loss_z_offset'] = loss_z_offset.item()
#         loc_loss += loss_z_offset
#
#         # y localization loss
#
#         y_offset_l, y_offset_r = start_offset, start_offset + 1
#         start_offset = y_offset_r
#
#         loss_y_offset = F.smooth_l1_loss(pred_ref[:, y_offset_l: y_offset_r].sum(dim=1),
#                                    y_offset_label - reg_label[:, 3] / 2)
#         reg_loss_dict['loss_y_offset'] = loss_y_offset.item()
#         loc_loss += loss_y_offset
#
#         # angle loss
#         ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
#         ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin
#
#         ry_label = reg_label[:, 6]
#
#         angle_per_class = (2 * np.pi) / num_head_bin
#         heading_angle = ry_label % (2 * np.pi)  # 0 ~ 2pi
#
#         shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
#         ry_bin_label = (shift_angle / angle_per_class).floor().long()
#         ry_res_label = shift_angle - (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
#         ry_res_norm_label = ry_res_label / (angle_per_class / 2)
#
#         ry_bin_onehot = torch.cuda.FloatTensor(ry_bin_label.size(0), num_head_bin).zero_()
#         ry_bin_onehot.scatter_(1, ry_bin_label.long().view(-1, 1), 1)
#         loss_ry_bin = F.cross_entropy(pred_ref[:, ry_bin_l:ry_bin_r], ry_bin_label)
#         loss_ry_res = F.smooth_l1_loss((pred_ref[:, ry_res_l: ry_res_r] * ry_bin_onehot).sum(dim=1), ry_res_norm_label)
#
#         reg_loss_dict['loss_ry_bin'] = loss_ry_bin.item()
#         reg_loss_dict['loss_ry_res'] = loss_ry_res.item()
#         angle_loss = loss_ry_bin + loss_ry_res
#
#         # size loss
#         size_res_l, size_res_r = ry_res_r, ry_res_r + 3
#
#         size_res_norm_label = (reg_label[:, 3:6] - anchor_size) / anchor_size
#         size_res_norm = pred_ref[:, size_res_l:size_res_r]
#         size_loss = F.smooth_l1_loss(size_res_norm, size_res_norm_label)
#
#         loc_loss = loc_loss
#         angle_loss = angle_loss
#         size_loss = size_loss
#         loss_iou_ref = loc_loss + angle_loss + size_loss
#     else:
#         loss_iou_ref = torch.zeros(()).cuda()
#     #
#     # iou_loss_dict['loss_iou_ref'] = loss_iou_ref.item()
#
#     return loss_iou_score, loss_iou_ref, iou_loss_dict
#     #return loss_iou_score, iou_loss_dict
#

