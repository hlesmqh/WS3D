import numpy as np
import os
import pickle
import torch
import copy
import random

from lib.datasets.kitti_dataset import KittiDataset
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.config import cfg
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from lib.utils.greedFurthestPoint import furthest_sample_pts
from matplotlib.lines import Line2D
from copy import deepcopy
#train500-car-exist-1085
import tqdm

class KittiBOXPLACEDataset():
    def __init__(self, root_dir, npoints=512, split='train', classes='Car', mode='TRAIN', random_select=True,
                 logger=None, noise=None, weakly_scene=100000,weakly_ratio=1.0):
        self.anchor_size = cfg.CLS_MEAN_SIZE
        self.anchor_max = np.array([[[2.0],[1.9],[5.0]]])
        self.anchor_min = np.array([[[1.2],[1.4],[2.6]]])

        self.split = split
        is_test = self.split == 'test'
        self.imageset_dir = os.path.join(root_dir, 'testing' if is_test else 'training')
        self.boxes_dir = os.path.join(self.imageset_dir, 'boxes_410fl030500_Car')
        if classes == 'Car':
            self.classes = ('Background', 'Car')
            aug_scene_root_dir = os.path.join(root_dir, 'KITTI', 'aug_scene')
        elif classes == 'People':
            self.classes = ('Background', 'Pedestrian', 'Cyclist')
        elif classes == 'Pedestrian':
            self.classes = ('Background', 'Pedestrian')
            aug_scene_root_dir = os.path.join(root_dir, 'KITTI', 'aug_scene_ped')
        elif classes == 'Cyclist':
            self.classes = ('Background', 'Cyclist')
            aug_scene_root_dir = os.path.join(root_dir, 'KITTI', 'aug_scene_cyclist')
        else:
            assert False, "Invalid classes: %s" % classes
        self.num_class = self.classes.__len__()

        if split=='train':
            df = open(os.path.join(self.boxes_dir, 'train_boxes.pkl'), 'rb')
            self.sample_id_list = pickle.load(df)

        elif split=='val':
            df = open(os.path.join(self.boxes_dir, 'val_boxes.pkl'), 'rb')
            self.sample_id_list = pickle.load(df)

        elif split=='small_val':
            df = open(os.path.join(self.boxes_dir, 'small_val_boxes.pkl'), 'rb')
            self.sample_id_list = pickle.load(df)

        else:
            NotImplementedError

        self.npoints = npoints
        self.random_select = random_select
        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode

        if self.mode == 'TRAIN':
            self.aug_id_list = []
            aug_num=4
        else:
            self.aug_id_list = []
            aug_num=1

        scene_list=[]
        instance_list=[]
        weakly_scene = weakly_scene

        self.feature_included=False

        for d in self.sample_id_list:

            d['aug_flag'] = 0
            d['sample_id'] = d['sample_id']
            d['box_id'] = d['box_id']
            d['center'] = d['center']
            # d['gt_boxes'][:,0] -= d['center'][:,0]
            # d['gt_boxes'][:,2] -= d['center'][:,2]
            d['foreground_flag'] = d['foreground_flag']
            d['cur_box_point'] = d['cur_box_point']
            d['cur_box_point'][:,1] = d['cur_box_point'][:,1]
            d['cur_box_reflect'] = d['cur_box_reflect'].reshape(-1, 1)
            if self.feature_included:
                d['cur_pts_feature'] = d['cur_pts_feature']
            elif 'cur_pts_feature' in d.keys():
                d.pop('cur_pts_feature')
            d['cur_prob_mask'] = (d['cur_prob_mask']>0.5).astype(np.float).reshape(-1, 1) - 0.5
            d['gt_mask'] = d['gt_mask'].reshape(-1, 1).astype(np.float) - 0.5
            instance_id = deepcopy([d['sample_id'], d['box_id']])
            if not instance_id in instance_list:
                instance_list.append(instance_id)
                if not instance_id[0] in scene_list:
                    scene_list.append(instance_id[0])

        weakly_instance_list=[]
        if self.mode=='TRAIN':
            if weakly_scene == 500:
                last_scene = 1085
            elif weakly_scene == 1632:
                last_scene = 3740
            elif weakly_scene > 3000:
                last_scene = 10000000
            else:
                NotImplementedError
            weakly_scene_list = [x for x in scene_list if x <= last_scene]

            # weakly_instance_list = [x for x in instance_list if ((x[0] in weakly_scene_list) and (x[1]>-1) and (not x in weakly_instance_list))]

            weakly_all_instance_list = [x for x in instance_list if ((x[0] in weakly_scene_list) and (x[1] > -1))]
            for id in weakly_all_instance_list:
                if id not in weakly_instance_list:
                    weakly_instance_list.append(id)
            random.seed(666)
            random.shuffle(weakly_instance_list)
            random.seed()
            weakly_instance_num = int(len(weakly_instance_list)*weakly_ratio)
            weakly_instance_list = weakly_instance_list[:weakly_instance_num]
        else:
            weakly_scene_list = scene_list
            weakly_instance_list = instance_list
        print('Loaded %d instance in %d scene.'%(len(weakly_instance_list),len(weakly_scene_list)))


        for i in range(aug_num):
            data = copy.deepcopy(self.sample_id_list)

            for d in data:
                if not d['sample_id'] in weakly_scene_list: continue
                if (not d['box_id']<0) and (not [d['sample_id'], d['box_id']] in weakly_instance_list): continue

                #only train TP
                #if  d['box_id'] < 0: continue


                # data input
                sample_id = d['sample_id']
                box_id = d['box_id']
                center = d['center']
                gt_boxes = d['gt_boxes']
                foreground_flag = d['foreground_flag']
                cur_box_point = d['cur_box_point']
                if self.feature_included:
                    cur_pts_feature = d['cur_pts_feature'].reshape(-1, 128)
                cur_box_reflect = d['cur_box_reflect'].reshape(-1, 1)
                cur_prob_mask = d['cur_prob_mask'].reshape(-1, 1)
                gt_mask = d['gt_mask'].reshape(-1, 1)

                if not self.mode == 'TRAIN':
                    gt_mask = cur_prob_mask

                d.update({
                               'aug_flag': i,
                               'sample_id': sample_id,
                               'box_id': box_id,
                               'center': center,
                               'gt_boxes': gt_boxes.reshape(7),
                               'foreground_flag': foreground_flag,
                               'cur_box_point': cur_box_point.reshape(-1, 3),
                               'cur_box_reflect': cur_box_reflect.reshape(-1, 1),
                               'cur_prob_mask': cur_prob_mask.reshape(-1, 1),
                               'gt_mask': gt_mask.reshape(-1, 1),
                               })
                if self.feature_included:
                    d.update({
                        'cur_pts_feature': cur_pts_feature.reshape(-1, 128),
                    })
                self.aug_id_list.append(d)

        self.sample_id_list = self.aug_id_list




        self.num_sample = self.sample_id_list.__len__()
        self.logger = logger


        # for rcnn training
        self.rpn_feature_list = {}
        self.pos_bbox_list = []
        self.neg_bbox_list = []
        self.far_neg_bbox_list = []

        self.gt_database = None

        if not self.random_select:
            self.logger.warning('random select is False')

        self.logger.info('Done: total samples %d' % self.num_sample)





    def __len__(self):
        if cfg.RCNN.ENABLED:
            return self.sample_id_list.__len__()
        elif cfg.IOUN.ENABLED:
            return self.sample_id_list.__len__()
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        #print(self.sample_id_list)
        return self.get_rcnn_sample(index)


    def get_rcnn_sample(self, index):

        data = copy.deepcopy(self.sample_id_list[index])
        if not self.mode == 'TEST':

            # data input
            aug_flag = data['aug_flag']
            sample_id = data['sample_id']
            box_id = data['box_id']
            center = data['center']
            gt_boxes = data['gt_boxes']
            foreground_flag = data['foreground_flag']
            if foreground_flag:
                cls = np.ones((1))
            else:
                cls = np.zeros((1))
            cur_box_point = data['cur_box_point']

            #ground point
            cur_box_point[:,1] = cur_box_point[:,1]-1.65
            gt_boxes[1] = gt_boxes[1]-1.65

            if self.feature_included:
                cur_pts_feature = data['cur_pts_feature']
            cur_box_reflect = data['cur_box_reflect'].reshape(-1, 1)
            cur_prob_mask = data['cur_prob_mask'].reshape(-1, 1)
            gt_mask = data['gt_mask'].reshape(-1, 1)
            if not self.split=='train':
                gt_mask = cur_prob_mask

            # network input sampling
            if self.random_select and self.mode == 'TRAIN':
                mask_noisy = np.random.uniform(0, 1, cur_prob_mask.shape[0])
                cur_prob_mask[mask_noisy > 0.95,:] = -cur_prob_mask[mask_noisy > 0.95,:]
                gt_mask[mask_noisy > 0.95,:] = - gt_mask[mask_noisy > 0.95,:]

                if self.feature_included:
                    pts_input = np.concatenate((cur_box_point, cur_box_reflect, cur_prob_mask, gt_mask, cur_pts_feature), axis=1)
                else:
                    pts_input = np.concatenate(
                        (cur_box_point, cur_box_reflect, cur_prob_mask, gt_mask), axis=1)

                np.random.shuffle(pts_input)

                # pts_input = pts_input[np.max(np.abs(pts_input[:,[0,2]]),axis=1)<3]

                # #Region dropout
                drop_out_random = np.random.uniform(-1, 1, 6)
                # # if drop_out_random[0] > 0.5:
                # #     if drop_out_random[1] > 0.5:
                # #         drop_out_index_x = np.logical_and(pts_input[:, 4] > 0, pts_input[:, 0] > gt_boxes[0])
                # #     else:
                # #         drop_out_index_x = np.logical_and(pts_input[:, 4] > 0, pts_input[:, 0] < gt_boxes[0])
                # #
                # #     if drop_out_random[2] > 0.5:
                # #         drop_out_index_z = np.logical_and(pts_input[:, 4] > 0, pts_input[:, 2] > gt_boxes[2])
                # #     else:
                # #         drop_out_index_z = np.logical_and(pts_input[:, 4] > 0, pts_input[:, 2] < gt_boxes[2])
                # #
                # #     if drop_out_random[0] > 0.75:
                # #         drop_out_index = np.logical_or(drop_out_index_x, drop_out_index_z)
                # #     else:
                # #         drop_out_index = np.logical_and(drop_out_index_x, drop_out_index_z)
                # #
                # #     if drop_out_random[4] > 0.5:
                # #         drop_out_index = np.logical_or(drop_out_index, pts_input[:, 4] < 0)
                # # else:
                # #     drop_out_index = pts_input[:, 4] > -1
                # #
                # # if np.max(np.logical_and(drop_out_index, pts_input[:, 5]>0)) == 0:
                # #     drop_out_index = pts_input[:, 4] > -1
                # #
                # # pts_input = pts_input[drop_out_index, :]

                # new drop out
                if drop_out_random[0] > 0.5:
                    if drop_out_random[1] > 0.0:
                        drop_out_index_x = np.logical_and(pts_input[:, 4] > 0, pts_input[:, 0] > gt_boxes[0])
                    else:
                        drop_out_index_x = np.logical_and(pts_input[:, 4] > 0, pts_input[:, 0] < gt_boxes[0])

                    if drop_out_random[2] > 0.5:
                        drop_out_index_z = np.logical_and(pts_input[:, 4] > 0, pts_input[:, 2] > gt_boxes[2])
                    else:
                        drop_out_index_z = np.logical_and(pts_input[:, 4] > 0, pts_input[:, 2] < gt_boxes[2])

                    if drop_out_random[5] > 0.0:
                        drop_out_index = np.logical_or(drop_out_index_x, drop_out_index_z)
                    else:
                        drop_out_index = np.logical_and(drop_out_index_x, drop_out_index_z)

                    if drop_out_random[4] > 0.5:
                        drop_out_index = np.logical_or(drop_out_index, pts_input[:, 4] < 0)
                else:
                    drop_out_index = pts_input[:, 4] > -1

                if np.max(np.logical_and(drop_out_index, pts_input[:, 5]>0)) == 0:
                    drop_out_index = pts_input[:, 4] > -1

                pts_input = pts_input[drop_out_index, :]


                # globaly random sampling
                # pts_input = pts_input[:min(pts_input.shape[0], self.npoints), :]
                # sample_index = np.arange(0, pts_input.shape[0], 1).astype(np.int)
                # perm = np.copy(sample_index)
                # while sample_index.shape[0] < self.npoints:
                #     sample_index = np.concatenate(
                #         (sample_index, perm[:min(perm.shape[0], self.npoints - sample_index.shape[0])]))

                #new sample method
                pts_input = pts_input[:min(pts_input.shape[0], self.npoints), :]
                if pts_input.shape[0]==512 and drop_out_random[3] > 0.5:
                    pts_input = pts_input[:128]
                    if drop_out_random[3] > 0.7:
                        pts_input = pts_input[:32]

                sample_index = np.arange(0, pts_input.shape[0], 1).astype(np.int)
                perm = np.copy(sample_index)
                while sample_index.shape[0] < self.npoints:
                    sample_index = np.concatenate(
                        (sample_index, perm[:min(perm.shape[0], self.npoints - sample_index.shape[0])]))

                cur_box_point = pts_input[sample_index, 0:3]
                cur_box_reflect = pts_input[sample_index, 3].reshape(-1, 1)
                cur_prob_mask = pts_input[sample_index, 4].reshape(-1, 1)
                gt_mask = pts_input[sample_index, 5].reshape(-1, 1)
                if self.feature_included:
                    cur_pts_feature = pts_input[sample_index, 6:].reshape(-1, 1)




            # rcnn noise adding
            #   generate noise
            noise = np.random.uniform(-1, 1, 6)
            if aug_flag == 0:
                noise = np.zeros(6)

            # #gaussian trans

            # g_noise = np.random.normal(0, 0.1, 2)
            # noise_x = g_noise[0]
            # noise_z = g_noise[1]
            # noise_y = noise[1] * 0.1

            #add y
            g_noise = np.random.normal(0, 0.1, 3)
            noise_x = g_noise[0]
            noise_z = g_noise[1]
            noise_y = noise[2]

            noise_filp = noise[5]
            noise_ry = noise[3] * np.pi / 2
            # gt scale noise gather

            # gaussian size
            noise[4] = np.random.normal(0, 0.1, 1)/2
            noise_scale = 1. + noise[4] * 0.20

            # extra size noise
            ext_noise = np.random.normal(0, 0.1, 3)
            ext_noise = 1. + ext_noise * 0.20
            revive_matrix = np.array([

                [[np.cos(-gt_boxes[6]), 0, np.sin(-gt_boxes[6]), 0],
                 [0, 1, 0, 0],
                 [-np.sin(-gt_boxes[6]), 0, np.cos(-gt_boxes[6]), 0],
                 [0, 0, 0, 1]],

                [[np.cos(gt_boxes[6]), 0, np.sin(gt_boxes[6]), 0],
                 [0, 1, 0, 0],
                 [-np.sin(gt_boxes[6]), 0, np.cos(gt_boxes[6]), 0],
                 [0, 0, 0, 1]]
            ])

            if not self.mode == 'TRAIN':
                noise_x = 0.0
                noise_y = 0.0
                noise_z = 0.0
                noise_ry = 0
                noise_scale = 1. + noise[4] * 0.
                ext_noise = 1. + ext_noise * 0.

            #noargue
            # noise_x = 0.0
            # noise_y = 0.0
            # noise_z = 0.0
            # noise_ry = 0
            # noise_filp = np.zeros(1)
            # noise_scale = np.ones(1)


            #  do transformation gt
            if foreground_flag:
                gt_boxes[6] = (gt_boxes[6] + noise_ry) % (2 * np.pi)
                if gt_boxes[6] > np.pi: gt_boxes[6] -= 2 * np.pi


            if noise_filp > 0:
                cur_box_point[:, 0] = -cur_box_point[:, 0]
                gt_boxes[0] = -gt_boxes[0]
                gt_boxes[6] = (np.pi - gt_boxes[6]) % (2 * np.pi)
                if gt_boxes[6] >= np.pi: gt_boxes[6] -= 2 * np.pi
                noise_ry = -noise_ry

            Rot_y = np.array(
                [[np.cos(noise_ry), 0, np.sin(noise_ry), noise_x],
                 [0, 1, 0, noise_y],
                 [-np.sin(noise_ry), 0, np.cos(noise_ry), noise_z],
                 [0, 0, 0, 1]])

            #  transform pointcloud
            cur_box_point = cur_box_point.reshape(-1, 3)
            # gt trans
            if aug_flag != 0 and self.mode == 'TRAIN':
                cur_box_point[:, 0] -= gt_boxes[0]
                cur_box_point[:, 2] -= gt_boxes[2]
                gt_boxes[0] = 0
                gt_boxes[2] = 0
            #basic trans
            cur_box_point = np.concatenate((cur_box_point, np.ones((cur_prob_mask.shape[0], 1))),
                                           axis=1)

            # cur_box_point = np.dot(Rot_y, cur_box_point.T).T[:, 0:3]
            # cur_box_point = cur_box_point.reshape(-1, 3)

            gt_boxes = gt_boxes.reshape(-1, 7)
            gt_boxes = np.concatenate((gt_boxes, np.ones((gt_boxes.shape[0], 1))),
                                      axis=1)


                # gt_boxes[:,0:3] = np.dot(Rot_y, gt_boxes_xyz1.T).T[:, 0:3]
                # gt_boxes = gt_boxes.reshape(7)

                # if self.mode == 'TRAIN':
                #     gt_boxes[3] = np.clip(gt_boxes[3] * noise_scale, self.anchor_min[:, 0], self.anchor_max[:, 0])
                #     gt_boxes[4] = np.clip(gt_boxes[4] * noise_scale, self.anchor_min[:, 1], self.anchor_max[:, 1])
                #     gt_boxes[5] = np.clip(gt_boxes[5] * noise_scale, self.anchor_min[:, 2], self.anchor_max[:, 2])
                # else:
                # gt_boxes[3] = gt_boxes[3] * noise_scale
                # gt_boxes[4] = gt_boxes[4] * noise_scale
                # gt_boxes[5] = gt_boxes[5] * noise_scale
                #


            # plot_gt_box = np.copy(gt_boxes[0,:7])
            #
            # fig, ax = plt.subplots(figsize=(5, 5))
            # # ax.axis([min(-35,center[0]), max(35,center[0]), min(0,center[1]), max(70,center[1])])
            # # plt.scatter(pts_rect[:, 0], pts_rect[:, 2], s=15, c=pts_rect[:, 1], edgecolor='none',
            # #             cmap=plt.get_cmap('Blues'), alpha=1, marker='.', vmin=0.0, vmax=1)
            # ax.axis([-4, 4, -4, 4])
            # plt.scatter(cur_box_point[:, 0], cur_box_point[:, 2], s=15, c=cur_prob_mask[:, 0], edgecolor='none',
            #             cmap=plt.get_cmap('rainbow'), alpha=1, marker='.', vmin=-0.5, vmax=0.5)
            #
            # pred_boxes3d_corner = kitti_utils.boxes3d_to_corners3d(plot_gt_box.reshape(1, 7), rotate=True)
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

            if not self.split=='train':
                gt_mask = cur_prob_mask

            if cfg.IOUN.ENABLED:
                iou_trans_list = []
                iou_scale_list = []
                iou_ry_list = []
                for i in range(cfg.CASCADE):
                    if self.mode=='TRAIN':
                        iou_noise = np.random.normal(0, 0.1, 6)*np.power(0.5,cfg.CASCADE-1)
                        iou_trans = iou_noise[0:3]
                        iou_trans[1] = iou_trans[1]
                        iou_scale = 1. + iou_noise[3] * 0.2
                        # iou_extraL = 1. + iou_noise[5] * 0.2
                        iou_ry = iou_noise[4] * np.pi / 10
                        # iou_noise = np.random.normal(0, 0.1, 5)
                        # iou_trans = iou_noise[0:3]
                        # iou_trans[1] = iou_trans[1]*0.3
                        # iou_scale = 1. + iou_noise[3] * 0.05
                        # iou_ry = iou_noise[4] * np.pi / 10

                    else:
                        iou_noise = np.zeros(6,dtype=float)
                        iou_trans = iou_noise[0:3]
                        iou_scale = 1. + iou_noise[3] * 0.2
                        # iou_extraL =  1. + iou_noise[5] * 0.2
                        iou_ry = iou_noise[4]

                    iou_trans_list.append(iou_trans.reshape(-1, 3,1))
                    iou_scale_list.append(iou_scale.reshape(-1, 1,1))
                    iou_ry_list.append(iou_ry.reshape(-1, 1,1))
                iou_trans = np.concatenate(iou_trans_list,axis=-1)
                iou_scale = np.concatenate(iou_scale_list, axis=-1)
                iou_ry = np.concatenate(iou_ry_list, axis=-1)


            if self.feature_included:
                sample_info = {'sample_id': sample_id,
                               'box_id': box_id,
                               'center': center,
                               'Rot_y': Rot_y.reshape(4,4),
                               'noise_scale': noise_scale.reshape(-1,1),
                               'gt_boxes': gt_boxes.reshape(1,8)*(cls),
                               'cls': cls.reshape(1),
                               'cur_box_point': cur_box_point.reshape(-1,4),
                               'cur_pts_feature': cur_pts_feature.reshape(-1,128),
                               'cur_box_reflect': cur_box_reflect.reshape(-1,1),
                               'cur_prob_mask': cur_prob_mask.reshape(-1,1),
                               'gt_mask': gt_mask.reshape(-1,1),
                               }
            else:
                if not cfg.IOUN.ENABLED:
                    sample_info = {'sample_id': sample_id,
                                   'box_id': box_id,
                                   'center': center,
                                   'Rot_y': Rot_y.reshape(4,4),
                                   'noise_scale': noise_scale.reshape(-1,1),
                                   'gt_boxes': gt_boxes.reshape(1,8)*(cls),
                                   'ext_noise': ext_noise.reshape(-1, 3),
                                   'revive_matrix': revive_matrix.reshape(2, 4, 4),
                                   'cls': cls.reshape(1),
                                   'cur_box_point': cur_box_point.reshape(-1,4),
                                   'cur_box_reflect': cur_box_reflect.reshape(-1,1),
                                   'cur_prob_mask': cur_prob_mask.reshape(-1,1),
                                   'gt_mask': gt_mask.reshape(-1,1),
                                   }
                else:
                    sample_info = {'sample_id': sample_id,
                                   'box_id': box_id,
                                   'center': center,
                                   'Rot_y': Rot_y.reshape(4, 4),
                                   'noise_scale': noise_scale.reshape(-1, 1),
                                   'iou_trans':iou_trans,
                                   'iou_scale': iou_scale,
                                   'iou_ry': iou_ry,
                                   'gt_boxes': gt_boxes.reshape(1, 8) * (cls),
                                   'ext_noise': ext_noise.reshape(-1, 3),
                                   'revive_matrix': revive_matrix.reshape(2, 4, 4),
                                   'cls': cls.reshape(1),
                                   'cur_box_point': cur_box_point.reshape(-1, 4),
                                   'cur_box_reflect': cur_box_reflect.reshape(-1, 1),
                                   'cur_prob_mask': cur_prob_mask.reshape(-1, 1),
                                   'gt_mask': gt_mask.reshape(-1, 1),
                                   }


            return sample_info

    @staticmethod
    def rotate_box3d_along_y(self, box3d, rot_angle):
        old_x, old_z, ry = box3d[0], box3d[2], box3d[6]
        old_beta = np.arctan2(old_z, old_x)
        alpha = -np.sign(old_beta) * np.pi / 2 + old_beta + ry

        box3d = kitti_utils.rotate_pc_along_y(box3d.reshape(1, 7), rot_angle=rot_angle)[0]
        new_x, new_z = box3d[0], box3d[2]
        new_beta = np.arctan2(new_z, new_x)
        box3d[6] = np.sign(new_beta) * np.pi / 2 + alpha - new_beta

        return box3d



    def collate_batch(self, batch):

        batch_size = batch.__len__()
        ans_dict = {}

        for key in batch[0].keys():
            #if key == 'box_id': continue
            if cfg.RPN.ENABLED and key == 'gt_boxes3d' or \
                    (cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, batch[k][key].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                for i in range(batch_size):
                    batch_gt_boxes3d[i, :batch[i][key].__len__(), :] = batch[i][key]
                ans_dict[key] = batch_gt_boxes3d
                continue

            if isinstance(batch[0][key], np.ndarray):
                if batch_size == 1:
                    ans_dict[key] = batch[0][key][np.newaxis, ...]
                else:
                    ans_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)], axis=0)

            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict


# plot_gt_box = np.copy(gt_boxes)
            #
            # fig, ax = plt.subplots(figsize=(5, 5))
            # # ax.axis([min(-35,center[0]), max(35,center[0]), min(0,center[1]), max(70,center[1])])
            # # plt.scatter(pts_rect[:, 0], pts_rect[:, 2], s=15, c=pts_rect[:, 1], edgecolor='none',
            # #             cmap=plt.get_cmap('Blues'), alpha=1, marker='.', vmin=0.0, vmax=1)
            # ax.axis([-4, 4, -4, 4])
            # plt.scatter(cur_box_point[:, 0], cur_box_point[:, 2], s=15, c=cur_prob_mask[:, 0] + 0.7, edgecolor='none',
            #             cmap=plt.get_cmap('Blues'), alpha=1, marker='.', vmin=0.0, vmax=1)
            #
            # pred_boxes3d_corner = kitti_utils.boxes3d_to_corners3d(plot_gt_box.reshape(1, 7), rotate=True)
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

if __name__ == '__main__':
    pass