import _init_path
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from torch_cluster import fps
from torch.utils.data import DataLoader
import torch.nn.functional as F
from lib.net.point_rcnn import PointRCNN
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from lib.utils.bbox_transform import decode_center_target, decode_bbox_target_stage_2
from lib.utils.kitti_utils import boxes3d_to_corners3d_torch
import tools.train_utils.train_utils as train_utils
from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate
import lib.utils.iou3d.iou3d_utils as iou3d_utils
from lib.utils.distance import distance_2, distance_2_numpy
import argparse
import lib.utils.kitti_utils as kitti_utils
from lib.utils.weighted_sample import  weighted_sample
import random
from datetime import datetime
import logging
import re
import glob
import time
from tensorboardX import SummaryWriter
import tqdm
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
matplotlib.use('agg')
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')
from sklearn.covariance import MinCovDet
from scipy.stats import multivariate_normal
from lib.utils.greedFurthestPoint import furthest_sample_pts
import shutil


np.random.seed(1024)  # set the same seed

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--cfg_file', type=str,                                     default='cfgs/',
                    help='specify the config for evaluation')

parser.add_argument('--eval_all', action='store_true',                          default=False,
                    help='whether to evaluate all checkpoints')
parser.add_argument('--test', action='store_true',                              default=False,
                    help='evaluate without ground truth')
parser.add_argument("--ckpt", type=str,                                         default=None,
                    help="specify a checkpoint to be evaluated")
parser.add_argument("--rpn_ckpt", type=str,                                     default=
        #'/raid/meng/Pointcloud_Detection/PointRCNN1.1_weak/output/rpn/weaklyRPN0500/103_th0.3_crowd_8000/ckpt/checkpoint_iter_06420.pth',
        #'/raid/meng/Pointcloud_Detection/PointRCNN1.1_weak/output/rpn/weaklyRPN0500/123_normalmask_8000/ckpt/checkpoint_iter_07620.pth',
        #'/raid/meng/Pointcloud_Detection/PointRCNN4_weak/output/rpn/weaklyRPN1632/413_floss03_8000/ckpt/checkpoint_iter_07995.pth',
        '/raid/meng/Pointcloud_Detection/PointRCNN4_weak/output/rpn/weaklyRPN0500/410_floss03_8000/ckpt/checkpoint_iter_07620.pth',
        #'/raid/meng/Pointcloud_Detection/PointRCNN4_weak/output/rpn/weaklyRPN3264/410_floss03_8000/ckpt/checkpoint_iter_07930.pth',
                    help="specify the checkpoint of rpn if trained separated")
parser.add_argument("--rcnn_ckpt", type=str,                                    default=
        #'/raid/meng/Pointcloud_Detection/PointRCNN1.1_weak/output/ioun/132_149_123_d03s01f02v_transall0.1g_WHL_refXXL_nocls_val_s500x0.25_10000/ckpt/checkpoint_iter_09936.pth',
        '/raid/meng/Pointcloud_Detection/PointRCNN5.1_weak/output/ioun/523_525_410_gpgr_rangeMSEpartreg_cascade1_s500x0.25_10000/ckpt/checkpoint_iter_09960.pth',
        #'/raid/meng/Pointcloud_Detection/PointRCNN_weak/output/ioun/41allscene_s1000000x0.25_80000/ckpt/checkpoint_iter_79940.pth',
                    help="specify the checkpoint of rcnn if trained separated")

parser.add_argument('--batch_size', type=int,                                   default=1,
                    help='batch size for evaluation')
parser.add_argument('--workers', type=int,                                      default=0,
                    help='number of workers for dataloader')
parser.add_argument("--extra_tag", type=str,                                    default='default',
                    help="extra tag for multiple evaluation")
parser.add_argument('--output_dir', type=str,                                   default=None,
                    help='specify an output directory if needed')
parser.add_argument("--ckpt_dir", type=str,                                     default=None,
                    help="specify a ckpt directory to be evaluated if needed")

parser.add_argument('--save_result', action='store_true',                       default=False,
                    help='save evaluation results to files')
parser.add_argument('--save_rpn_feature', action='store_true',                  default=False,
                    help='save features for separately rcnn training and evaluation')

parser.add_argument('--random_select', action='store_true',                     default=False,
                    help='sample to the same number of points')
parser.add_argument('--start_epoch', type=int,                                  default=0,
                    help='ignore the checkpoint smaller than this epoch')
parser.add_argument("--rcnn_eval_roi_dir", type=str,                            default=None,
                    help='specify the saved rois for rcnn evaluation when using rcnn_offline mode')
parser.add_argument("--rcnn_eval_feature_dir", type=str,                        default=None,
                    help='specify the saved features for rcnn evaluation when using rcnn_offline mode')
parser.add_argument('--set', dest='set_cfgs',                                   default=None,
                    nargs=argparse.REMAINDER, help='set extra config keys if needed')
args = parser.parse_args()
#if DEBUG
VISUAL=False
#endif
def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def save_kitti_format(sample_id, calib, bbox3d, kitti_output_dir, scores, img_shape):
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

    kitti_output_file = os.path.join(kitti_output_dir, '%06d.txt' % sample_id)
    with open(kitti_output_file, 'w') as f:
        for k in range(bbox3d.shape[0]):
            if box_valid_mask[k] == 0:
                continue
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
                  (cfg.CLASSES, alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
                   bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                   bbox3d[k, 6], scores[k]), file=f)



def eval_one_epoch_joint(model, dataloader, epoch_id, result_dir, logger):
    np.random.seed(666)
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
    mode = 'TEST' if args.test else 'EVAL'

    final_output_dir = os.path.join(result_dir, 'final_result', 'data')

    if os.path.exists(final_output_dir): shutil.rmtree(final_output_dir)
    os.makedirs(final_output_dir, exist_ok=True)

    logger.info('---- EPOCH %s JOINT EVALUATION ----' % epoch_id)
    logger.info('==> Output file: %s' % result_dir)
    model.eval()

    thresh_list = [0.1,0.3,0.5,0.7,0.9]
    total_recalled_bbox_list, total_gt_bbox = [0] * 5, 0
    total_roi_recalled_bbox_list = [0] * 5
    dataset = dataloader.dataset
    cnt = final_total = total_cls_acc = total_cls_acc_refined = total_rpn_iou = 0
    obj_num = 0
    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval')

    iou_list = []
    iou_p_score_list = []
    rcnn_p_score_list = []
    prop_count = 0
    for data in dataloader:


        # Loading sample
        sample_id_list, pts_input = data['sample_id'], data['pts_input']
        sample_id = sample_id_list[0]
        cnt += len(sample_id_list)
        #if cnt < 118: continue
        #load label
        if not args.test:
            gt_boxes3d = data['gt_boxes3d']
            obj_num += gt_boxes3d.shape[1]
            # print(obj_num)
            if gt_boxes3d.shape[1] == 0:  # (B, M, 7)
                pass
            else:
                gt_boxes3d = gt_boxes3d

        # rpn model inference
        inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
        #inputs = inputs[:,torch.argsort(-inputs[0,:,2])]
        input_data = {'pts_input': inputs}
        ret_dict = model.rpn_forward(input_data)
        rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']
        rpn_backbone_xyz, rpn_backbone_features = ret_dict['backbone_xyz'], ret_dict['backbone_features']

        # stage score parsing
        rpn_scores_raw = rpn_cls[:, :, 0]
        rpn_scores_norm = torch.sigmoid(rpn_cls[:, :, 0])
        rcnn_input_scores = rpn_scores_norm.view(-1).clone()
        inputs = inputs.view(-1, inputs.shape[-1])
        rpn_backbone_features = rpn_backbone_features.view(-1, rpn_backbone_features.shape[-2])
        rpn_backbone_xyz = rpn_backbone_xyz.view(-1, rpn_backbone_xyz.shape[-1])


        # if VISUAL:
        #     order = torch.argsort(-rpn_scores_norm).view(-1)
        #     inputs = inputs.view(-1,inputs.shape[-1])[order]
        #     rpn_scores_norm = rpn_scores_norm.view(-1)[order]
        #     rpn_backbone_features = rpn_backbone_features.view(-1,rpn_backbone_features.shape[-1])[order]
        #
        #     norm_feature = F.normalize(rpn_backbone_features)
        #     similarity = norm_feature.mm(norm_feature.t())
        #
        #     inputs_plt = inputs.detach().cpu().numpy()
        #     scores_plt = rpn_scores_norm.detach().cpu().numpy()
        #     similarity_plt = similarity.detach().cpu().numpy()
        #
        #
        #     fig = plt.figure(figsize=(10, 10))
        #     plt.axes(facecolor='silver')
        #     plt.axis([-30,30,0,70])
        #     plt.title('point_regressed_center %06d'%sample_id)
        #     plt.scatter(inputs_plt[:, 0], inputs_plt[:, 2], s=15, c=scores_plt[:], edgecolor='none',
        #                 cmap=plt.get_cmap('rainbow'), alpha=1, marker='.', vmin=0, vmax=1)
        #     if args.test==False:
        #         gt_boxes3d = gt_boxes3d.reshape(-1,7)
        #         plt.scatter(gt_boxes3d[:, 0], gt_boxes3d[:, 2], s=200, c='blue',
        #                     alpha=0.5, marker='+', vmin=-1, vmax=1)
        #     plt.show()
        #
        #     for i in range(similarity_plt.shape[0]):
        #         fig = plt.figure(figsize=(10, 10))
        #         plt.axes(facecolor='silver')
        #         plt.axis([-30, 30, 0, 70])
        #         sm_plt = similarity_plt[i]
        #         plt.scatter(inputs_plt[i, 0].reshape(-1), inputs_plt[i, 2].reshape(-1), s=400, c='blue',
        #                     alpha=0.5, marker='+', vmin=-1, vmax=1)
        #         plt.scatter(inputs_plt[:, 0], inputs_plt[:, 2], s=15, c=(sm_plt[:]+scores_plt[:])/2, edgecolor='none',
        #                     cmap=plt.get_cmap('rainbow'), alpha=1, marker='.', vmin=0, vmax=1)
        #         plt.show()


        # thresh select and jump out
        # rpn_mask = rpn_scores_norm.view(-1) > cfg.RPN.SCORE_THRESH
        # if rpn_mask.float().sum() == 0: continue
        # rpn_scores_raw = rpn_scores_raw.view(-1)[rpn_mask]
        # rpn_scores_norm = rpn_scores_norm.view(-1)[rpn_mask]
        # rpn_reg = rpn_reg.view(-1, rpn_reg.shape[-1])[rpn_mask]
        # rpn_backbone_xyz = rpn_backbone_xyz.view(-1, rpn_backbone_xyz.shape[-1])[rpn_mask]

        # generate rois


        rpn_rois = decode_center_target(rpn_backbone_xyz, rpn_reg.view(-1, rpn_reg.shape[-1]),
                                        loc_scope=cfg.RPN.LOC_SCOPE,
                                        loc_bin_size=cfg.RPN.LOC_BIN_SIZE,
                                        ).view(-1, 3)
        rpn_reg_dist = (rpn_rois - rpn_backbone_xyz).clone()
        #similarity = torch.cosine_similarity(rpn_backbone_xyz[:, [0, 2]], rpn_reg_dist[:, [0, 2]], dim=1)

        # # thresh select and jump out
        rpn_mask = (rpn_scores_norm.view(-1) > cfg.RPN.SCORE_THRESH) & (rpn_reg_dist[:,[0,2]].pow(2).sum(-1).sqrt()>0.2) #\
                   #& (similarity > -0.7)
        if rpn_mask.float().sum() == 0: continue
        rpn_scores_raw = rpn_scores_raw.view(-1)[rpn_mask]
        rpn_scores_norm = rpn_scores_norm.view(-1)[rpn_mask]
        rpn_rois = rpn_rois[rpn_mask]
        rpn_backbone_xyz = rpn_backbone_xyz.view(-1, rpn_backbone_xyz.shape[-1])[rpn_mask]

        # radius NMS
        # sort by center score
        sort_points = torch.argsort(-rpn_scores_raw)
        rpn_rois = rpn_rois[sort_points]
        rpn_scores_norm = rpn_scores_norm[sort_points]
        rpn_scores_raw = rpn_scores_raw[sort_points]

        if rpn_rois.shape[0] > 1:
            keep_id = [0]
            prop_prop_distance = distance_2(rpn_rois[:, [0, 2]], rpn_rois[:, [0, 2]])
            for i in range(1, rpn_rois.shape[0]):
                #if torch.min(prop_prop_distance[:i, i], dim=-1)[0] > 0.3:
                if torch.min(prop_prop_distance[keep_id, i], dim=-1)[0] > 0.3:
                    keep_id.append(i)
            rpn_center = rpn_rois[keep_id][:,[0,2]]
            rpn_scores_norm = rpn_scores_norm[keep_id]
            rpn_scores_raw = rpn_scores_raw[keep_id]

        else:
            rpn_center = rpn_rois[:, [0, 2]]
            rpn_scores_norm = rpn_scores_norm
            rpn_scores_raw = rpn_scores_raw

        # #rcnn input select:
        point_center_distance = distance_2(rpn_center, inputs[:,[0,2]])
        cur_proposal_points_index = (torch.min(point_center_distance, dim=-1)[0] < 4.0)

        point_center_distance = point_center_distance[cur_proposal_points_index]
        inputs = inputs[cur_proposal_points_index]
        rcnn_input_scores = rcnn_input_scores.view(-1)[cur_proposal_points_index]


        if VISUAL:
            inputs_plt = inputs.detach().cpu().numpy()
            scores_plt = rcnn_input_scores.detach().cpu().numpy()
            # point_center= rpn_center[rpn_scores_norm > 0.5]
            # point_center_score = rpn_scores_norm[rpn_scores_norm > 0.5]
            point_center= rpn_center
            point_center_score = rpn_scores_norm
            fig = plt.figure(figsize=(10, 10))
            plt.axes(facecolor='silver')
            plt.axis([-30,30,0,70])
            point_center_plt = point_center.cpu().numpy()
            plt.title('point_regressed_center %06d'%sample_id)
            plt.scatter(inputs_plt[:, 0], inputs_plt[:, 2], s=15, c=scores_plt[:], edgecolor='none',
                        cmap=plt.get_cmap('rainbow'), alpha=1, marker='.', vmin=0, vmax=1)
            if point_center.shape[0] > 0:
                plt.scatter(point_center_plt[:, 0], point_center_plt[:, 1], s=200, c='white',
                            alpha=0.5, marker='x', vmin=-1, vmax=1)
            if args.test==False:
                gt_boxes3d = gt_boxes3d.reshape(-1,7)
                plt.scatter(gt_boxes3d[:, 0], gt_boxes3d[:, 2], s=200, c='blue',
                            alpha=0.5, marker='+', vmin=-1, vmax=1)
            plt.savefig('../visual/rpn.jpg')


        # RCNN stage
        box_list = []
        raw_score_list = []
        iou_score_list = []
        inputs[:, 1] -= 1.65
        point_center_distance = distance_2(rpn_center[:, :], inputs[:, [0, 2]])
        #for c in range(min(rpn_center.shape[0],100)):
        prop_count += rpn_center.shape[0]
        print('num %d'%(prop_count/float(cnt)))
        for c in range(rpn_center.shape[0]):
            # rcnn input generate
            cur_input = inputs.clone()
            cur_input_score = rcnn_input_scores.clone()

            # if COSINE_DISTANCE:
            #     cur_center_points_index = ((point_center_distance[:, c] < 4.0) & \
            #                                (point_prop_cos_matrix[:, c] > COS_THRESH) | \
            #                                (point_center_distance[:, c].view(-1) < 0.7)).view(-1)
            # else:
            cur_center_points_index = (point_center_distance[:, c] < 4.0).view(-1)
            if cur_center_points_index.long().sum() == 0: continue

            cur_center_points_xyz = cur_input[cur_center_points_index, :3]
            cur_center_points_xyz[:, 0] -= rpn_center[c, 0]
            cur_center_points_xyz[:, 2] -= rpn_center[c, 1]
            cur_center_points_r = cur_input[cur_center_points_index, 3].view(-1, 1)
            cur_center_points_mask = (cur_input_score[cur_center_points_index] > 0.5).view(-1, 1).float()

            # # easy sample sampling
            # if pts_input.shape[0]>512:
            #     cur_input = torch.cat((cur_center_points_xyz, cur_center_points_r,
            #                            (cur_input_score[cur_center_points_index] > 0.5).view(-1, 1).float()), dim=-1)
            #     pts_input = cur_input
            #     pts_input = pts_input[:min(pts_input.shape[0], 2000), :]
            #     pts_input = pts_input[:, :]
            #     sample_index = fps(pts_input[:, 0:3].contiguous(), ratio=min(512 / pts_input.shape[0], 0.99),
            #                        random_start=False)
            #     perm = sample_index
            #     while sample_index.shape[0] < 512:
            #         sample_index = torch.cat(
            #             (sample_index, perm[:min(perm.shape[0], 512 - sample_index.shape[0])]), dim=0)
            #
            #     cur_center_points_xyz = pts_input[sample_index, 0:3]
            #     cur_center_points_r = pts_input[sample_index, 3].reshape(-1, 1)
            #     cur_center_points_mask = pts_input[sample_index, 4].reshape(-1, 1)

            cur_center_points_xyz = cur_center_points_xyz.unsqueeze(0).float()
            cur_center_points_r = cur_center_points_r.unsqueeze(0).float()
            cur_center_points_mask = cur_center_points_mask.unsqueeze(0).float() - 0.5

            input_data = {'cur_box_point': cur_center_points_xyz,
                          'cur_box_reflect': cur_center_points_r,
                          'train_mask': cur_center_points_mask,
                          }

            # # globaly random sampling
            # pts_input = pts_input[:min(pts_input.shape[0], self.npoints), :]
            # sample_index = np.arange(0, pts_input.shape[0], 1).astype(np.int)
            # perm = np.copy(sample_index)
            # while sample_index.shape[0] < self.npoints:
            #     sample_index = np.concatenate(
            #         (sample_index, perm[:min(perm.shape[0], self.npoints - sample_index.shape[0])]))
            #
            # cur_box_point = pts_input[sample_index, 0:3]
            # cur_box_reflect = pts_input[sample_index, 3].reshape(-1, 1)
            # cur_prob_mask = pts_input[sample_index, 4].reshape(-1, 1)
            # gt_mask = pts_input[sample_index, 5].reshape(-1, 1)

            # rcnn model inference
            ret_dict = model.rcnn_forward(input_data)
            rcnn_cls = ret_dict['rcnn_cls']
            ioun_cls = ret_dict['ioun_cls']
            rcnn_reg = ret_dict['rcnn_reg']
            rcnn_iou = ret_dict['rcnn_iou']
            rcnn_ref = ret_dict['rcnn_ref'].view(1,1,-1)
            rcnn_box3d = ret_dict['pred_boxes3d']
            refined_box = ret_dict['refined_box']

            rcnn_box3d = refined_box
            rcnn_box3d[:,:,6] = rcnn_box3d[:,:,6]%(np.pi*2)
            if rcnn_box3d[:, :, 6]>np.pi: rcnn_box3d[:,:,6] -= np.pi * 2

            rcnn_box3d[:, :, 0] += rpn_center[c][0]
            rcnn_box3d[:, :, 2] += rpn_center[c][1]
            rcnn_box3d[:, :, 1] += 1.65

            box_list.append(rcnn_box3d)

            raw_score_list.append(rcnn_cls.view(1,1))
            #raw_score_list.append(ioun_cls.view(1,1))

            iou_score_list.append(rcnn_iou.view(1,1))

        rcnn_box3d = torch.cat((box_list), dim=1)
        raw_rcnn_score = torch.cat((raw_score_list), dim=0).unsqueeze(0).float()
        norm_ioun_score = torch.cat((iou_score_list), dim=0).unsqueeze(0).float()

        # scoring
        pred_boxes3d = rcnn_box3d
        norm_ioun_score = norm_ioun_score
        raw_rcnn_score = raw_rcnn_score
        norm_rcnn_score = torch.sigmoid(raw_rcnn_score)

        # scores thresh
        pred_h = pred_boxes3d[:,:,3].view(-1)
        pred_w = pred_boxes3d[:,:,4].view(-1)
        pred_l = pred_boxes3d[:,:,5].view(-1)
        inds = (norm_rcnn_score > cfg.RCNN.SCORE_THRESH) & (norm_ioun_score > cfg.IOUN.SCORE_THRESH)
        inds = inds.view(-1)
        #size filiter
        # inds = inds & \
        #         (pred_h > 1.2) & (pred_h < 2.2) & \
        #         (pred_w > 1.3) & (pred_w < 2.0) & \
        #         (pred_l > 2.2) & (pred_l < 5.0)
        inds = inds & \
                (pred_h > 1.1) & (pred_h < 2.3) & \
                (pred_w > 1.2) & (pred_w < 2.1) & \
                (pred_l > 2.1) & (pred_l < 5.1)


        pred_boxes3d = pred_boxes3d[:,inds]
        norm_rcnn_score = norm_rcnn_score[:,inds]
        norm_ioun_score = norm_ioun_score[:,inds]
        raw_rcnn_score = raw_rcnn_score[:,inds]

        if pred_boxes3d.shape[1] == 0: continue
        # evaluation
        recalled_num = gt_num = 0

        if not args.test:
            gt_boxes3d = data['gt_boxes3d']

            for k in range(1):
                # calculate recall
                cur_gt_boxes3d = gt_boxes3d[k]
                tmp_idx = cur_gt_boxes3d.__len__() - 1

                while tmp_idx >= 0 and cur_gt_boxes3d[tmp_idx].sum() == 0:
                    tmp_idx -= 1

                if tmp_idx >= 0:
                    cur_gt_boxes3d = cur_gt_boxes3d[:tmp_idx + 1]

                    cur_gt_boxes3d = torch.from_numpy(cur_gt_boxes3d).cuda(non_blocking=True).float()
                    _, iou3d = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d[k], cur_gt_boxes3d)
                    gt_max_iou, _ = iou3d.max(dim=0)
                    refined_iou, _ = iou3d.max(dim=1)

                    iou_list.append(refined_iou.view(-1,1))
                    iou_p_score_list.append(norm_ioun_score.view(-1,1))
                    rcnn_p_score_list.append(norm_rcnn_score.view(-1,1))

                    for idx, thresh in enumerate(thresh_list):
                        total_recalled_bbox_list[idx] += (gt_max_iou > thresh).sum().item()
                    recalled_num += (gt_max_iou > 0.7).sum().item()
                    gt_num += cur_gt_boxes3d.shape[0]
                    total_gt_bbox += cur_gt_boxes3d.shape[0]

        if cnt == 1000:
            iou_clloe = torch.cat(iou_list, dim=0).detach().cpu().numpy()
            iou_score_clloe = torch.cat(iou_p_score_list, dim=0).detach().cpu().numpy()
            plt.axis([-.1, 1.1, -.1, 1.1])
            plt.scatter(iou_clloe, iou_score_clloe, s=20, c='blue', edgecolor='none', cmap=plt.get_cmap('YlOrRd'),
                        alpha=1,
                        marker='.')
            plt.savefig(os.path.join(result_dir, 'distributercnn.png'))

        disp_dict = {'mode': mode, 'recall': '%d/%d' % (total_recalled_bbox_list[3], total_gt_bbox)}
        progress_bar.set_postfix(disp_dict)
        progress_bar.update()


        if VISUAL:
            fig, ax = plt.subplots(figsize=(10, 10))
            inputs_plt = inputs.detach().cpu().numpy()
            #plt.axes(facecolor='silver')
            plt.axis([-35, 35, 0, 70])
            plt.scatter(inputs_plt[:, 0], inputs_plt[:, 2], s=15, c=inputs_plt[:, 1],
                        edgecolor='none',
                        cmap=plt.get_cmap('Blues'), alpha=1, marker='.', vmin=-1, vmax=2)
            pred_boxes3d_numpy = pred_boxes3d[0].detach().cpu().numpy()
            pred_boxes3d_corner = kitti_utils.boxes3d_to_corners3d(pred_boxes3d_numpy, rotate=True)
            for o in range(pred_boxes3d_corner.shape[0]):
                print_box_corner = pred_boxes3d_corner[o]

                x1, x2, x3, x4 = print_box_corner[0:4, 0]
                z1, z2, z3, z4 = print_box_corner[0:4, 2]

                polygon = np.zeros([5, 2], dtype=np.float32)
                polygon[0, 0] = x1
                polygon[1, 0] = x2
                polygon[2, 0] = x3
                polygon[3, 0] = x4
                polygon[4, 0] = x1

                polygon[0, 1] = z1
                polygon[1, 1] = z2
                polygon[2, 1] = z3
                polygon[3, 1] = z4
                polygon[4, 1] = z1

                line1 = [(x1, z1), (x2, z2)]
                line2 = [(x2, z2), (x3, z3)]
                line3 = [(x3, z3), (x4, z4)]
                line4 = [(x4, z4), (x1, z1)]
                (line1_xs, line1_ys) = zip(*line1)
                (line2_xs, line2_ys) = zip(*line2)
                (line3_xs, line3_ys) = zip(*line3)
                (line4_xs, line4_ys) = zip(*line4)
                ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='green'))
                ax.add_line(Line2D(line2_xs, line2_ys, linewidth=1, color='red'))
                ax.add_line(Line2D(line3_xs, line3_ys, linewidth=1, color='red'))
                ax.add_line(Line2D(line4_xs, line4_ys, linewidth=1, color='red'))

                # gt visualize

            if args.test==False and data['gt_boxes3d'].shape[1] > 0:
                gt_boxes3d_corner = kitti_utils.boxes3d_to_corners3d(data['gt_boxes3d'].reshape(-1, 7), rotate=True)

                for o in range(gt_boxes3d_corner.shape[0]):
                    print_box_corner = gt_boxes3d_corner[o]

                    x1, x2, x3, x4 = print_box_corner[0:4, 0]
                    z1, z2, z3, z4 = print_box_corner[0:4, 2]

                    polygon = np.zeros([5, 2], dtype=np.float32)
                    polygon[0, 0] = x1
                    polygon[1, 0] = x2
                    polygon[2, 0] = x3
                    polygon[3, 0] = x4
                    polygon[4, 0] = x1

                    polygon[0, 1] = z1
                    polygon[1, 1] = z2
                    polygon[2, 1] = z3
                    polygon[3, 1] = z4
                    polygon[4, 1] = z1

                    line1 = [(x1, z1), (x2, z2)]
                    line2 = [(x2, z2), (x3, z3)]
                    line3 = [(x3, z3), (x4, z4)]
                    line4 = [(x4, z4), (x1, z1)]
                    (line1_xs, line1_ys) = zip(*line1)
                    (line2_xs, line2_ys) = zip(*line2)
                    (line3_xs, line3_ys) = zip(*line3)
                    (line4_xs, line4_ys) = zip(*line4)
                    ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='yellow'))
                    ax.add_line(Line2D(line2_xs, line2_ys, linewidth=1, color='purple'))
                    ax.add_line(Line2D(line3_xs, line3_ys, linewidth=1, color='purple'))
                    ax.add_line(Line2D(line4_xs, line4_ys, linewidth=1, color='purple'))
            plt.savefig('../visual/rcnn.jpg')


        # scores thresh
        inds =  (norm_rcnn_score > cfg.RCNN.SCORE_THRESH) & (norm_ioun_score > cfg.IOUN.SCORE_THRESH)
        #inds = (norm_ioun_score > cfg.IOUN.SCORE_THRESH)

        for k in range(1):
            cur_inds = inds[k].view(-1)
            if cur_inds.sum() == 0:
                continue

            pred_boxes3d_selected = pred_boxes3d[k, cur_inds]
            norm_iou_scores_selected = norm_ioun_score[k, cur_inds]
            raw_rcnn_score_selected = raw_rcnn_score[k, cur_inds]

            #traditional nms
            # NMS thresh rotated nms
            # boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(pred_boxes3d_selected)
            # #score NMS
            # # boxes_bev_selected[:,-1] += np.pi/2
            # keep_idx = iou3d_utils.nms_normal_gpu(boxes_bev_selected, norm_iou_scores_selected, cfg.RCNN.NMS_THRESH).view(-1)
            # pred_boxes3d_selected = pred_boxes3d_selected[keep_idx]
            # norm_iou_scores_selected = norm_iou_scores_selected[keep_idx]
            # raw_rcnn_score_selected = raw_rcnn_score_selected[keep_idx]


            #self NMS
            sort_boxes = torch.argsort(-norm_iou_scores_selected.view(-1))
            pred_boxes3d_selected = pred_boxes3d_selected[sort_boxes]
            norm_iou_scores_selected = norm_iou_scores_selected[sort_boxes]

            if pred_boxes3d_selected.shape[0] > 1:
                keep_id = [0]
                iou2d, iou3d = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d_selected, pred_boxes3d_selected)
                for i in range(1, pred_boxes3d_selected.shape[0]):
                    # if torch.min(prop_prop_distance[:i, i], dim=-1)[0] > 0.3:
                    if torch.max(iou2d[keep_id, i], dim=-1)[0] < 0.01:
                        keep_id.append(i)
                pred_boxes3d_selected = pred_boxes3d_selected[keep_id]
                norm_iou_scores_selected = norm_iou_scores_selected[keep_id]
            else:
                pred_boxes3d_selected = pred_boxes3d_selected
                norm_iou_scores_selected = norm_iou_scores_selected

            pred_boxes3d_selected, norm_iou_scores_selected = pred_boxes3d_selected.cpu().numpy(), norm_iou_scores_selected.cpu().numpy()

            cur_sample_id = sample_id
            calib = dataset.get_calib(cur_sample_id)
            final_total += pred_boxes3d_selected.shape[0]
            image_shape = dataset.get_image_shape(cur_sample_id)
            save_kitti_format(cur_sample_id, calib, pred_boxes3d_selected, final_output_dir, norm_iou_scores_selected, image_shape)

            if VISUAL:
                fig, ax = plt.subplots(figsize=(10, 10))
                inputs_plt = inputs.detach().cpu().numpy()
                # plt.axes(facecolor='silver')
                plt.axis([-35, 35, 0, 70])
                plt.scatter(inputs_plt[:, 0], inputs_plt[:, 2], s=15, c=inputs_plt[:, 1],
                            edgecolor='none',
                            cmap=plt.get_cmap('Blues'), alpha=1, marker='.', vmin=-1, vmax=2)
                pred_boxes3d_numpy = pred_boxes3d_selected
                pred_boxes3d_corner = kitti_utils.boxes3d_to_corners3d(pred_boxes3d_numpy, rotate=True)
                for o in range(pred_boxes3d_corner.shape[0]):
                    print_box_corner = pred_boxes3d_corner[o]

                    x1, x2, x3, x4 = print_box_corner[0:4, 0]
                    z1, z2, z3, z4 = print_box_corner[0:4, 2]

                    polygon = np.zeros([5, 2], dtype=np.float32)
                    polygon[0, 0] = x1
                    polygon[1, 0] = x2
                    polygon[2, 0] = x3
                    polygon[3, 0] = x4
                    polygon[4, 0] = x1

                    polygon[0, 1] = z1
                    polygon[1, 1] = z2
                    polygon[2, 1] = z3
                    polygon[3, 1] = z4
                    polygon[4, 1] = z1

                    line1 = [(x1, z1), (x2, z2)]
                    line2 = [(x2, z2), (x3, z3)]
                    line3 = [(x3, z3), (x4, z4)]
                    line4 = [(x4, z4), (x1, z1)]
                    (line1_xs, line1_ys) = zip(*line1)
                    (line2_xs, line2_ys) = zip(*line2)
                    (line3_xs, line3_ys) = zip(*line3)
                    (line4_xs, line4_ys) = zip(*line4)
                    ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='green'))
                    ax.add_line(Line2D(line2_xs, line2_ys, linewidth=1, color='red'))
                    ax.add_line(Line2D(line3_xs, line3_ys, linewidth=1, color='red'))
                    ax.add_line(Line2D(line4_xs, line4_ys, linewidth=1, color='red'))

                    # gt visualize

                if args.test == False and data['gt_boxes3d'].shape[1] > 0:
                    gt_boxes3d_corner = kitti_utils.boxes3d_to_corners3d(data['gt_boxes3d'].reshape(-1, 7), rotate=True)

                    for o in range(gt_boxes3d_corner.shape[0]):
                        print_box_corner = gt_boxes3d_corner[o]

                        x1, x2, x3, x4 = print_box_corner[0:4, 0]
                        z1, z2, z3, z4 = print_box_corner[0:4, 2]

                        polygon = np.zeros([5, 2], dtype=np.float32)
                        polygon[0, 0] = x1
                        polygon[1, 0] = x2
                        polygon[2, 0] = x3
                        polygon[3, 0] = x4
                        polygon[4, 0] = x1

                        polygon[0, 1] = z1
                        polygon[1, 1] = z2
                        polygon[2, 1] = z3
                        polygon[3, 1] = z4
                        polygon[4, 1] = z1

                        line1 = [(x1, z1), (x2, z2)]
                        line2 = [(x2, z2), (x3, z3)]
                        line3 = [(x3, z3), (x4, z4)]
                        line4 = [(x4, z4), (x1, z1)]
                        (line1_xs, line1_ys) = zip(*line1)
                        (line2_xs, line2_ys) = zip(*line2)
                        (line3_xs, line3_ys) = zip(*line3)
                        (line4_xs, line4_ys) = zip(*line4)
                        ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='yellow'))
                        ax.add_line(Line2D(line2_xs, line2_ys, linewidth=1, color='purple'))
                        ax.add_line(Line2D(line3_xs, line3_ys, linewidth=1, color='purple'))
                        ax.add_line(Line2D(line4_xs, line4_ys, linewidth=1, color='purple'))
                plt.savefig('../visual/ioun.jpg')

    progress_bar.close()
    # dump empty files
    split_file = os.path.join(dataset.imageset_dir, '..', 'ImageSets', dataset.split + '.txt')
    split_file = os.path.abspath(split_file)
    image_idx_list = [x.strip() for x in open(split_file).readlines()]
    empty_cnt = 0
    for k in range(image_idx_list.__len__()):
        cur_file = os.path.join(final_output_dir, '%s.txt' % image_idx_list[k])
        if not os.path.exists(cur_file):
            with open(cur_file, 'w') as temp_f:
                pass
            empty_cnt += 1
            logger.info('empty_cnt=%d: dump empty file %s' % (empty_cnt, cur_file))

    ret_dict = {'empty_cnt': empty_cnt}


    if not args.eval_all:
        logger.info('-------------------performance of epoch %s---------------------' % epoch_id)
        logger.info(str(datetime.now()))

        avg_rpn_iou = (total_rpn_iou / max(cnt, 1.0))
        avg_cls_acc = (total_cls_acc / max(cnt, 1.0))
        avg_cls_acc_refined = (total_cls_acc_refined / max(cnt, 1.0))
        avg_det_num = (final_total / max(len(dataset), 1.0))
        logger.info('final average detections: %.3f' % avg_det_num)
        logger.info('final average rpn_iou refined: %.3f' % avg_rpn_iou)
        logger.info('final average cls acc: %.3f' % avg_cls_acc)
        logger.info('final average cls acc refined: %.3f' % avg_cls_acc_refined)
        ret_dict['rpn_iou'] = avg_rpn_iou
        ret_dict['rcnn_cls_acc'] = avg_cls_acc
        ret_dict['rcnn_cls_acc_refined'] = avg_cls_acc_refined
        ret_dict['rcnn_avg_num'] = avg_det_num

        for idx, thresh in enumerate(thresh_list):
            cur_recall = total_recalled_bbox_list[idx] / max(total_gt_bbox, 1.0)
            logger.info('total bbox recall(thresh=%.3f): %d / %d = %f' % (thresh, total_recalled_bbox_list[idx],
                                                                          total_gt_bbox, cur_recall))
            ret_dict['rcnn_recall(thresh=%.2f)' % thresh] = cur_recall
            if thresh == 0.7:
                recall = cur_recall

    if cfg.TEST.SPLIT != 'test':
        logger.info('Averate Precision:')
        name_to_class = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        ap_result_str, ap_dict = kitti_evaluate(dataset.label_dir, final_output_dir, label_split_file=split_file,
                                                current_class=name_to_class[cfg.CLASSES])
        if not args.eval_all:
            logger.info(ap_result_str)
            ret_dict.update(ap_dict)

    logger.info('result is saved to: %s' % result_dir)
    precision=ap_dict['Car_3d_easy'] + ap_dict['Car_3d_moderate'] + ap_dict['Car_3d_hard']
    recall = total_recalled_bbox_list[3] / max(total_gt_bbox, 1.0)
    F2_score=0
    return precision,recall,F2_score


# def eval_one_epoch(model, dataloader, epoch_id, result_dir, logger):
#
#     #F2_score = eval_one_epoch_rcnn(model, dataloader, epoch_id, result_dir, logger)
#     F2_score = eval_one_epoch_joint(model, dataloader, epoch_id, result_dir, logger)
#     return F2_score


def load_part_ckpt(model, filename, logger, total_keys=-1):
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


def load_ckpt_based_on_args(model, logger):

    rpn_keys = model.rpn.state_dict().keys().__len__()
    rcnn_keys = model.rcnn_net.state_dict().keys().__len__()
    if cfg.RPN.ENABLED and args.rpn_ckpt is not None:
        load_part_ckpt(model, filename=args.rpn_ckpt, logger=logger, total_keys=rpn_keys)
    if cfg.RCNN.ENABLED and args.rcnn_ckpt is not None:
        load_part_ckpt(model, filename=args.rcnn_ckpt, logger=logger, total_keys=rcnn_keys)

def eval_single_ckpt(root_result_dir):
    root_result_dir = os.path.join(root_result_dir, 'eval')
    # set epoch_id and output dir
    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    iter_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    root_result_dir = os.path.join(root_result_dir, 'epoch_%s' % iter_id, cfg.TEST.SPLIT)
    if args.test:
        root_result_dir = os.path.join(root_result_dir, 'test_mode')

    if args.extra_tag != 'default':
        root_result_dir = os.path.join(root_result_dir, args.extra_tag)
    os.makedirs(root_result_dir, exist_ok=True)

    log_file = os.path.join(root_result_dir, 'log_eval_one.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    save_config_to_file(cfg, logger=logger)

    # create dataloader & network
    test_loader = create_dataloader(logger)
    model = PointRCNN(num_classes=test_loader.dataset.num_class, use_xyz=True, mode='TEST')

    model.cuda()

    # copy important files to backup
    backup_dir = os.path.join(root_result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp *.py %s/' % backup_dir)
    os.system('cp ../lib/net/*.py %s/' % backup_dir)
    os.system('cp ../lib/datasets/kitti_rcnn_dataset.py %s/' % backup_dir)

    # load checkpoint
    load_ckpt_based_on_args(model, logger)

    # start evaluation
    eval_one_epoch_joint(model, test_loader, iter_id, root_result_dir, logger)


def eval_all_ckpt(root_result_dir):
    root_result_dir = os.path.join('/'.join(args.rcnn_ckpt.split('/')[:-1]), 'all', 'eval')
    os.makedirs(root_result_dir, exist_ok=True)
    # set epoch_id and output dir
    ckpt_dir = '/'.join(args.rcnn_ckpt.split('/')[:-1])
    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list = [x for x in ckpt_list if x[-4:] == '.pth']
    ckpt_list.sort()
    BEST_precision = 0.
    BEST_iter = None
    log_file = os.path.join(root_result_dir, 'log_eval_all.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    save_config_to_file(cfg, logger=logger)

    # create dataloader & network
    test_loader = create_dataloader(logger)
    model = PointRCNN(num_classes=test_loader.dataset.num_class, use_xyz=True, mode='TEST')
    model.cuda()

    for ckpt in tqdm.tqdm(reversed(ckpt_list[25:])):
        args.rcnn_ckpt = os.path.join(ckpt_dir, ckpt)
        num_list = re.findall(r'\d+', args.rcnn_ckpt) if args.rcnn_ckpt is not None else []
        iter_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'

        cur_root_result_dir = os.path.join(root_result_dir, cfg.TEST.SPLIT)
        if args.test:
            cur_root_result_dir = os.path.join(root_result_dir, 'test_mode')

        if args.extra_tag != 'default':
            cur_root_result_dir = os.path.join(cur_root_result_dir, args.extra_tag)
        os.makedirs(cur_root_result_dir, exist_ok=True)

        # load checkpoint
        load_ckpt_based_on_args(model, logger)

        precision, _, _ = eval_one_epoch_joint(model, test_loader, iter_id, cur_root_result_dir, logger)
        if precision > BEST_precision:
            BEST_precision = precision
            BEST_iter = iter_id
        print('best_precision: %.4f, best_iter: %s,' % (BEST_precision, BEST_iter))
        print(args.rcnn_ckpt[-4:])


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def create_dataloader(logger):
    mode = 'TEST' if args.test else 'EVAL'
    DATA_PATH = os.path.join('/raid/meng/Dataset/Kitti/object')
    if args.eval_all:
        print('Args eval_all enabled, small_val set will be used')
        cfg.TEST.SPLIT = 'small_val'

    # create dataloader
    test_set = KittiRCNNDataset(root_dir=DATA_PATH, npoints=cfg.RPN.NUM_POINTS, split=cfg.TEST.SPLIT, mode=mode,
                                random_select=args.random_select,
                                classes=cfg.CLASSES,
                                logger=logger)#,noise='label_noise')

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.workers, collate_fn=test_set.collate_batch)

    return test_loader


if __name__ == "__main__":
    # merge config and log to file
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file + 'weaklyRPN.yaml')
        cfg_from_file(args.cfg_file + 'weaklyRCNN.yaml')
        cfg_from_file(args.cfg_file + 'weaklyIOUN.yaml')
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    cfg.RCNN.ENABLED = True
    cfg.RPN.ENABLED = cfg.RPN.FIXED = True
    cfg.IOUN.ENABLED = True

    # root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG + 'place_l2')
    root_result_dir = os.path.join(args.rcnn_ckpt[:-4]+'3.31')
    if args.output_dir is not None:
        root_result_dir = args.output_dir

    os.makedirs(root_result_dir, exist_ok=True)

    if args.eval_all:
        with torch.no_grad():
            eval_all_ckpt(root_result_dir)
    else:
        with torch.no_grad():
            eval_single_ckpt(root_result_dir)