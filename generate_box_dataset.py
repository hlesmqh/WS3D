#import _init_path
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import numpy as np
import pickle
import torch
import logging
from torch.utils.data import DataLoader
from lib.datasets.kitti_boxgen_dataset import KittiDataset
import lib.utils.kitti_utils as kitti_utils
from lib.utils.distance import distance_2, distance_2_numpy, cos_distance, cos_matrix_distance
from torch_cluster import fps
from lib.config import cfg, cfg_from_file, save_config_to_file
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lib.utils.bbox_transform import decode_center_target


import warnings
warnings.filterwarnings('ignore')


split='small_val'
if split == 'train':
    dataset = KittiDataset('/raid/meng/Dataset/Kitti/object', split=split,noise='label_noise')
else:
    dataset = KittiDataset('/raid/meng/Dataset/Kitti/object', split=split)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)
save_dir = '/raid/meng/Dataset/Kitti/object/training/boxes_dataset'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
ckpt_file = '/raid/meng/Pointcloud_Detection/PointRCNN4_weak/output/rpn/weaklyRPN0500/410_floss03_8000/ckpt/checkpoint_iter_07620.pth'
cfg_from_file('/raid/meng/Pointcloud_Detection/PointRCNN1.1_weak/tools/cfgs/weaklyRPN.yaml')

cfg.RPN.SCORE_THRESH = 0.1
PROP_DIST = 0.3
BACKGROUND_ADDING = False
BACK_THRESH = 0.3
COSINE_DISTANCE = False
COS_THRESH = 0.3

from lib.net.point_rcnn import PointRCNN
model = PointRCNN(num_classes=data_loader.dataset.num_class, use_xyz=True, mode='TEST')
model.cuda()
checkpoint = torch.load(ckpt_file)
model_state = checkpoint['model_state']

update_model_state = {key: val for key, val in model_state.items() if key in model.state_dict()}
state_dict = model.state_dict()
state_dict.update(update_model_state)
model.load_state_dict(state_dict)

update_keys = update_model_state.keys().__len__()
model.eval()


max_point=0
instance_id = 0
fg_sum=0
Gfg_sum=0
bg_sum=0
recall_count = 0
gt_count = 0
ALL_database = []
for data in dataset:
    # data loading
    sample_id = data['sample_id']
    if sample_id > 1085:
        break
    #if sample_id<40: continue
    image = data['image'][:,:,[2,1,0]]
    calib = data['calib']
    pts_lidar = data['pts_lidar']
    pts_rect = data['pts_rect']
    pts_reflect = data['pts_reflect']
    pts_image = data['pts_image']
    pts_input = np.concatenate((pts_rect, pts_reflect.reshape(-1, 1)), axis=1)[np.newaxis,:]
    if not split == 'test':
        gt_boxes3d_cam = data['gt_boxes_3d_cam']
        gt_boxes2d_cam = data['gt_boxes2d_cam']
        noise_gt_boxes3d_cam = data['noise_gt_boxes3d_cam']
        gt_alpha = data['gt_alpha']

    # model inference
    inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
    #inputs = inputs[:, torch.argsort(-inputs[0, :, 2])]
    input_data = {'pts_input': inputs}
    ret_dict = model.rpn_forward(input_data)
    rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']
    rpn_backbone_xyz, rpn_backbone_features = ret_dict['backbone_xyz'], ret_dict['backbone_features']

    # stage score parsing
    rpn_scores_raw = rpn_cls.view(-1, 1)
    rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
    rpn_input_scores = rpn_scores_norm.clone()
    rpn_backbone_xyz = rpn_backbone_xyz.view(-1, rpn_backbone_xyz.shape[-1])
    rpn_backbone_features = rpn_backbone_features.view(-1, rpn_backbone_features.shape[-2])

    inputs = inputs.view(-1, 4)

    # generate rois
    rpn_rois = decode_center_target(rpn_backbone_xyz, rpn_reg.view(-1, rpn_reg.shape[-1]),
                                    loc_scope=cfg.RPN.LOC_SCOPE,
                                    loc_bin_size=cfg.RPN.LOC_BIN_SIZE,
                                    ).view(-1, 3)

    rpn_reg_dist = (rpn_rois - rpn_backbone_xyz).clone()
    #similarity = torch.cosine_similarity(rpn_backbone_xyz[:,[0,2]], rpn_reg_dist[:,[0,2]], dim=1)
    #similarity = similarity.detach().cpu().numpy()
    # Thresh select
    rpn_mask = (rpn_scores_norm.view(-1) > cfg.RPN.SCORE_THRESH) & (rpn_reg_dist[:,[0,2]].pow(2).sum(-1).sqrt()>0.2) #\
                    # & (similarity > -0.7)
    if rpn_mask.float().sum() == 0: continue
    rpn_rois = rpn_rois[rpn_mask]
    rpn_scores_raw = rpn_scores_raw.view(-1)[rpn_mask]
    rpn_scores_norm = rpn_scores_norm.view(-1)[rpn_mask]
    rpn_reg = rpn_reg.view(-1, rpn_reg.shape[-1])[rpn_mask]
    rpn_backbone_xyz = rpn_backbone_xyz[rpn_mask]

    # radius NMS
    # sort by center score
    sort_points = torch.argsort(-rpn_scores_norm)
    rpn_rois = rpn_rois[sort_points]
    rpn_scores_norm = rpn_scores_norm[sort_points]
    rpn_scores_raw = rpn_scores_raw[sort_points]

    if rpn_rois.shape[0] > 1:
        keep_id = [0]
        prop_prop_distance = distance_2(rpn_rois[:, [0, 2]], rpn_rois[:, [0, 2]])
        for i in range(1, rpn_rois.shape[0]):
            # if torch.min(prop_prop_distance[:i, i], dim=-1)[0] > 0.3:
            if torch.min(prop_prop_distance[keep_id, i], dim=-1)[0] > PROP_DIST:
                keep_id.append(i)
        rpn_pred_center = rpn_rois[keep_id]
        rpn_scores_norm = rpn_scores_norm[keep_id]
        rpn_scores_raw = rpn_scores_raw[keep_id]
    else:
        rpn_pred_center = rpn_rois
        rpn_scores_norm = rpn_scores_norm
        rpn_scores_raw = rpn_scores_raw

    # # # # # todo visualization cluster distance estimation
    # inputs = inputs.detach().cpu().numpy()
    # rpn_cls = rpn_cls.detach().cpu().numpy()
    #
    # point_center = rpn_pred_center
    # point_center_score = rpn_scores_norm
    # fig = plt.figure(figsize=(10, 10))
    # plt.axes(facecolor='silver')
    # plt.axis([-30,30,0,70])
    # point_center_plt = point_center.cpu().numpy()
    # plt.title('point_regressed_center %06d'%sample_id)
    # plt.scatter(inputs[:, 0], inputs[:, 2], s=15, c=inputs[:, 1], edgecolor='none',
    #             cmap=plt.get_cmap('Blues'), alpha=1, marker='.', vmin=-1, vmax=2)
    # if point_center.shape[0] > 0:
    #     plt.scatter(point_center_plt[:, 0], point_center_plt[:, 2], s=200, c='white',
    #                 alpha=0.5, marker='x', vmin=-1, vmax=1)
    # if not split == 'test':
    #     plt.scatter(gt_boxes3d_cam[:, 0], gt_boxes3d_cam[:, 2], s=200, c='blue',
    #                 alpha=0.5, marker='+', vmin=-1, vmax=1)
    # plt.show()
    # continue

    if not split == 'test':
        if gt_boxes3d_cam.shape[0]==0:
            foreground_flag = torch.zeros(rpn_scores_norm.shape[0]).cuda()>0
            foreground_flag_G = torch.zeros(rpn_scores_norm.shape[0]).cuda() > 0
        else:
            gt_boxes3d_cam = torch.from_numpy(gt_boxes3d_cam).cuda().view(-1, 7)
            noise_gt_boxes3d_cam = torch.from_numpy(noise_gt_boxes3d_cam).cuda().view(-1, 7)

            proposal_noise_gt_distance = distance_2(noise_gt_boxes3d_cam[:,[0,2]],rpn_pred_center[:,[0,2]])
            proposal_gt_distance = distance_2(gt_boxes3d_cam[:,[0,2]],rpn_pred_center[:,[0,2]])

            proposal_gt_index = torch.argmin(proposal_gt_distance, dim=-1)
            if split=='train':
                foreground_flag = (torch.min(proposal_gt_distance, dim=-1)[0] < 0.7) | \
                                  (torch.min(proposal_noise_gt_distance, dim=-1)[0] < 0.7)
                foreground_flag_G = (torch.min(proposal_gt_distance, dim=-1)[0] < 1.5) | \
                                  (torch.min(proposal_noise_gt_distance, dim=-1)[0] < 1.5)
            else:
                foreground_flag = torch.min(proposal_gt_distance, dim=-1)[0] < 0.7
                foreground_flag_G = torch.min(proposal_gt_distance, dim=-1)[0] < 1.5


            recall_count += (torch.min(proposal_noise_gt_distance, dim=0)[0] <0.7).float().sum()
            gt_count += proposal_noise_gt_distance.shape[1]

    # background cancelation
    # if BACKGROUND_CANCEL:
    #     cancel_mask = rpn_input_scores.view(-1)>BACK_THRESH
    #     inputs = inputs[cancel_mask]
    #     rpn_reg_dist = rpn_reg_dist[cancel_mask]
    #     rpn_input_scores = rpn_input_scores[cancel_mask]
    #     rpn_backbone_features = rpn_backbone_features[cancel_mask]

    if COSINE_DISTANCE:
        point_prop_cos_matrix = cos_matrix_distance(rpn_pred_center[:, [0, 2]],inputs[:,[0,2]],rpn_reg_dist[:,[0,2]])
    # parsing to instance
    point_proposal_distance = distance_2(rpn_pred_center[:, [0, 2]], inputs[:, [0, 2]])


    for i in range(rpn_pred_center.shape[0]):
        if COSINE_DISTANCE: # cosine
            if BACKGROUND_ADDING:
                point_instance_flag = (point_proposal_distance[:,i].view(-1)<4.0)&\
                                      (point_prop_cos_matrix[:,i]>COS_THRESH) |\
                                      (point_proposal_distance[:,i].view(-1)<0.7) |\
                                      ((rpn_input_scores.view(-1)<BACK_THRESH)&\
                                       (point_proposal_distance[:,i].view(-1)<4.0))
            else:
                point_instance_flag = (point_proposal_distance[:,i].view(-1)<4.0)&\
                                      (point_prop_cos_matrix[:,i]>COS_THRESH) |\
                                      (point_proposal_distance[:,i].view(-1)<0.7)
        else: #normal
            point_instance_flag = (point_proposal_distance[:, i].view(-1) < 4.0)

        if point_instance_flag.long().sum()==0:continue

        cur_pts_rect =(inputs)[point_instance_flag,:3].detach().cpu().numpy()
        cur_box_reflect = (inputs)[point_instance_flag, -1].detach().cpu().numpy()
        cur_pts_mask = (rpn_input_scores)[point_instance_flag].detach().cpu().numpy()
        cur_pts_feature = (rpn_backbone_features)[point_instance_flag].detach().cpu().numpy()
        cur_pts_center = rpn_pred_center[i].detach().cpu().numpy()

        cur_box_point = (cur_pts_rect[:,:3] - cur_pts_center[:3].reshape(1,3))
        cur_prob_mask = cur_pts_mask.reshape(-1,1)
        cur_pts_feature = cur_pts_feature.reshape(-1,cur_pts_feature.shape[-1])
        cur_pts_center = cur_pts_center.reshape(3)

        if split == 'test': continue
        box_id = -1
        fg_flag = False
        gt_box = np.zeros(7)
        gt_mask = np.zeros((cur_prob_mask.shape))
        if foreground_flag[i]==True:
            fg_flag = True
        if foreground_flag_G[i] == True:
            box_id = proposal_gt_index[i].detach().cpu().numpy()
            gt_box = gt_boxes3d_cam[box_id].detach().cpu().numpy().reshape(7)
            gt_box[0] = gt_box[0] - cur_pts_center[0]
            gt_box[2] = gt_box[2] - cur_pts_center[2]
            gt_box[3] = gt_box[3] * 1.2
            gt_box[4] = gt_box[4] * 1.2
            gt_box[5] = gt_box[5] * 1.2
            gt_corners = kitti_utils.boxes3d_to_corners3d(gt_box.reshape(-1,7), rotate=True)
            gt_mask = kitti_utils.in_hull(cur_box_point, gt_corners.reshape(-1,3)).reshape(-1,1)

            gt_box = gt_boxes3d_cam[box_id].detach().cpu().numpy().reshape(7)
            gt_box[0] = gt_box[0] - cur_pts_center[0]
            gt_box[2] = gt_box[2] - cur_pts_center[2]

        # fig, ax = plt.subplots(figsize=(5, 5))
        # plt.title('%d / %d'%(box_id, gt_boxes3d_cam.shape[0]))
        # ax.axis([-4, 4, -4, 4])
        # plt.scatter(cur_box_point[:, 0], cur_box_point[:, 2], s=15, c=cur_prob_mask[:,0], edgecolor='none',
        #             cmap=plt.get_cmap('rainbow'), alpha=1, marker='.', vmin=0, vmax=1)
        # plt.scatter(np.zeros(1), np.zeros(1), s=200, c='black',
        #             alpha=0.5, marker='x', vmin=-1, vmax=1)
        # if foreground_flag[i] == True:
        #     plt.scatter(gt_box[0], gt_box[2], s=200, c='blue',
        #                 alpha=0.5, marker='+', vmin=-1, vmax=1)
        #     gt_corners = kitti_utils.boxes3d_to_corners3d(gt_box.reshape(-1, 7), rotate=True)
        #     pred_boxes3d_corner = gt_corners
        #     print_box_corner = pred_boxes3d_corner[0]
        #     x1, x2, x3, x4 = print_box_corner[0:4, 0]
        #     z1, z2, z3, z4 = print_box_corner[0:4, 2]
        #     polygon = np.zeros([5, 2], dtype=np.float32)
        #     polygon[0, 0] = x1
        #     polygon[1, 0] = x2
        #     polygon[2, 0] = x3
        #     polygon[3, 0] = x4
        #     polygon[4, 0] = x1
        #     polygon[0, 1] = z1
        #     polygon[1, 1] = z2
        #     polygon[2, 1] = z3
        #     polygon[3, 1] = z4
        #     polygon[4, 1] = z1
        #     line1 = [(x1, z1), (x2, z2)]
        #     line2 = [(x2, z2), (x3, z3)]
        #     line3 = [(x3, z3), (x4, z4)]
        #     line4 = [(x4, z4), (x1, z1)]
        #     (line1_xs, line1_ys) = zip(*line1)
        #     (line2_xs, line2_ys) = zip(*line2)
        #     (line3_xs, line3_ys) = zip(*line3)
        #     (line4_xs, line4_ys) = zip(*line4)
        #     ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='green'))
        #     ax.add_line(Line2D(line2_xs, line2_ys, linewidth=1, color='red'))
        #     ax.add_line(Line2D(line3_xs, line3_ys, linewidth=1, color='red'))
        #     ax.add_line(Line2D(line4_xs, line4_ys, linewidth=1, color='red'))
        # plt.show()

        if (not split=='train') or (split=='train' and cur_box_point.shape[0]>5):
            gt_database = {'instance_id': instance_id,
                           'sample_id': sample_id,
                           'box_id': int(box_id),
                           'center': cur_pts_center.reshape(1,3),
                           'foreground_flag': fg_flag,
                           'gt_boxes': gt_box.reshape(1,7),
                           'cur_box_point': cur_box_point,
                           'cur_box_reflect': cur_box_reflect.reshape(-1,1),
                           #'cur_pts_feature': cur_pts_feature,
                           'cur_prob_mask': cur_prob_mask,
                           'gt_mask': gt_mask,
                           }

            ALL_database.append(gt_database)
            instance_id += 1
            fg_sum += foreground_flag[i].long().cpu().numpy()
            Gfg_sum += foreground_flag_G[i].long().cpu().numpy()
            bg_sum += (1 - foreground_flag[i].long()).cpu().numpy()
            print('sample %s, instance %d, fg %d, bg %d, Gfg %d, recall %.4f' % (sample_id, instance_id, fg_sum, bg_sum, Gfg_sum, recall_count/max(gt_count,1)))


save_file_name = os.path.join(save_dir, '%s_boxes.pkl'%split)
print('Writing to file, please wait')
with open(save_file_name, 'wb') as f:
    # dict_data = {
    #             'sample_list': dataset.image_idx_list,
    #             'ALL_database': ALL_database,
    #             }
    pickle.dump(ALL_database, f)
print('Done')
print(cfg.RPN.SCORE_THRESH,PROP_DIST,BACKGROUND_ADDING,BACK_THRESH,COSINE_DISTANCE,COS_THRESH)


