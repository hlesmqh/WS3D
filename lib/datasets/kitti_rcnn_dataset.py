import numpy as np
import os
import pickle
import torch

from lib.datasets.kitti_dataset import KittiDataset
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.config import cfg
from scipy.stats import multivariate_normal
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import copy
from lib.utils.distance import distance_2, distance_2_numpy
from pointnet2_lib.pointnet2 import pointnet2_utils
import math
HARD_MIMIC_NUM = 128
GT_DATABASE_SPARSE_DISTANCE = 6.0
AUG_NUM=15

class KittiRCNNDataset(KittiDataset):
    def __init__(self, root_dir, npoints=16384, split='train', classes='Car', mode='TRAIN', random_select=True,
                 logger=None, noise=None, weakly_num=3265):
        super().__init__(root_dir=root_dir, split=split, noise=noise)

        if classes == 'Car':
            self.classes = ('Background', 'Car')
        elif classes == 'People':
            self.classes = ('Background', 'Pedestrian', 'Cyclist')
        elif classes == 'Pedestrian':
            self.classes = ('Background', 'Pedestrian')
        elif classes == 'Cyclist':
            self.classes = ('Background', 'Cyclist')
        else:
            assert False, "Invalid classes: %s" % classes

        self.num_class = self.classes.__len__()

        self.npoints = npoints
        self.sample_id_list = []
        self.random_select = random_select
        self.logger = logger

        # for rcnn training
        self.rpn_feature_list = {}
        self.pos_bbox_list = []
        self.neg_bbox_list = []
        self.far_neg_bbox_list = []

        if not self.random_select:
            self.logger.warning('random select is False')

        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode

        if mode == 'TRAIN':
            #loading samples
            self.logger.info('Loading %s samples from %s ...' % (self.mode, self.noise_label_dir))
            for idx in range(0, self.num_sample):
                sample_id = int(self.image_idx_list[idx])
                obj_list = self.filtrate_objects(self.get_noise_label(sample_id))
                if len(obj_list) == 0:
                    # self.logger.info('No gt classes: %06d' % sample_id)
                    continue
                self.sample_id_list.append(sample_id)
            self.logger.info('Done: filter %s results: %d / %d\n' % (self.mode, len(self.sample_id_list),
                                                                     len(self.image_idx_list)))
            self.sample_id_list = self.sample_id_list[:weakly_num]
            self.logger.info('Done: selection %s results: %d - %s\n' % (self.mode, len(self.sample_id_list), self.sample_id_list[-1]))

            # loading augment gts
            self.logger.info('Loading %s samples from %s ...' % (self.mode, self.noise_label_dir))
            if cfg.GT_AUG_ENABLED:
                df = open(os.path.join(self.imageset_dir, 'aug_gt_database.pkl'), 'rb')
                self.gt_database = pickle.load(df)
                self.gt_database = [gt for gt in self.gt_database if int(gt['sample_id'])<=int(self.sample_id_list[-1])]
                self.logger.info('Done: selection %s gt: %d\n' % (self.mode, len(self.gt_database)))

                if cfg.GT_AUG_HARD_RATIO > 0:
                    easy_list, hard_list = [], []
                    for gt in self.gt_database:
                        if gt['presampling_flag']:
                            easy_list.append(gt)
                        else:
                            hard_list.append(gt)
                    self.gt_database = [easy_list, hard_list]
                logger.info('Loading gt_database(easy(pt_num>512): %d, hard(pt_num<=512): %d) from aug_gt_database'
                            % (len(easy_list), len(hard_list)))
        else:
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
            self.logger.info('Load testing samples from %s' % self.imageset_dir)
            self.logger.info('Done: total test samples %d' % len(self.sample_id_list))


    #
    # def get_road_plane(self, idx):
    #     return super().get_road_plane(idx % 10000)

    # @staticmethod
    # def get_rpn_features(rpn_feature_dir, idx):
    #     rpn_feature_file = os.path.join(rpn_feature_dir, '%06d.npy' % idx)
    #     rpn_xyz_file = os.path.join(rpn_feature_dir, '%06d_xyz.npy' % idx)
    #     rpn_intensity_file = os.path.join(rpn_feature_dir, '%06d_intensity.npy' % idx)
    #     if cfg.RCNN.USE_SEG_SCORE:
    #         rpn_seg_file = os.path.join(rpn_feature_dir, '%06d_rawscore.npy' % idx)
    #         rpn_seg_score = np.load(rpn_seg_file).reshape(-1)
    #         rpn_seg_score = torch.sigmoid(torch.from_numpy(rpn_seg_score)).numpy()
    #     else:
    #         rpn_seg_file = os.path.join(rpn_feature_dir, '%06d_seg.npy' % idx)
    #         rpn_seg_score = np.load(rpn_seg_file).reshape(-1)
    #     return np.load(rpn_xyz_file), np.load(rpn_feature_file), np.load(rpn_intensity_file).reshape(-1), rpn_seg_score

    def filtrate_objects(self, obj_list):
        """
        Discard objects which are not in self.classes (or its similar classes)
        :param obj_list: list
        :return: list
        """
        type_whitelist = self.classes
        if self.mode == 'TRAIN' and cfg.INCLUDE_SIMILAR_TYPE:
            type_whitelist = list(self.classes)
            if 'Car' in self.classes:
                type_whitelist.append('Van')
            if 'Pedestrian' in self.classes:  # or 'Cyclist' in self.classes:
                type_whitelist.append('Person_sitting')

        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in type_whitelist:  # rm Van, 20180928
                continue
            if self.mode == 'TRAIN' and cfg.PC_REDUCE_BY_RANGE and (self.check_pc_range(obj.pos) is False):
                continue
            valid_obj_list.append(obj)
        return valid_obj_list

    @staticmethod
    def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        """
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param pts_img:
        :param pts_rect_depth:
        :param img_shape:
        :return:
        """
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        if cfg.PC_REDUCE_BY_RANGE:
            x_range, y_range, z_range = cfg.PC_AREA_SCOPE
            pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
            range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
            pts_valid_flag = pts_valid_flag & range_flag
        return pts_valid_flag

    @staticmethod
    def check_pc_range(xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range, y_range, z_range = cfg.PC_AREA_SCOPE
        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False


    # def adding_aug_boxs(self,pts_rect, pts_intensity, gt_centers):
    #
    #     # random select sample and copy
    #     add_gts = []
    #     assert AUG_NUM%3==0
    #     add_gts += copy.deepcopy(random.sample(self.gt_database[0], int(AUG_NUM/3*2)))
    #     add_gts += copy.deepcopy(random.sample(self.gt_database[1], int(AUG_NUM/3)))
    #     #5 easy 5 mimic hard 5 real hard
    #     for i in range(int(AUG_NUM/3),int(AUG_NUM/3*2)):
    #         #mimic hard sample
    #         if add_gts[i]['presampling_flag']:
    #             cur_add_inputs = add_gts[i]['aug_gt_input']
    #             cur_add_inputs = cur_add_inputs[add_gts[i]['sampled_flag']]
    #             cur_add_inputs = np.random.choice(cur_add_inputs, HARD_MIMIC_NUM, replace=False)
    #             add_gts[i]['aug_gt_input'] = cur_add_inputs
    #
    #     #generate in sphere range and change them to xz
    #     add_center_ceta = np.random.rand(0.25*np.pi,0.75*np.pi,AUG_NUM)
    #     add_center_depth = np.concatenate((np.random.rand(40.,70.,int(AUG_NUM/3*2)),np.rand.random(3,40.,int(AUG_NUM/3))))
    #     add_center = np.zeros((AUG_NUM,3))
    #     add_center[:, 0] = np.cos(add_center_ceta) * add_center_depth
    #     add_center[:, 2] = np.sin(add_center_ceta) * add_center_depth
    #
    #     # collided detection
    #     #cat aug with original
    #     gt_aug_centers = np.concatenate((gt_centers,add_center),axis=0)
    #     distance_gt_matrix = distance_2_numpy(gt_aug_centers[:,[0,2]],add_center[:,[0,2]])
    #     keep_id = []
    #     ori_gt_num = gt_centers.shape[0]
    #     for i in range(AUG_NUM):
    #         if np.min(distance_gt_matrix[i,:(i+ori_gt_num)]) > GT_DATABASE_SPARSE_DISTANCE:
    #             keep_id.append(i)
    #
    #     add_gts = add_gts[keep_id]
    #     add_center = add_gts[keep_id]
    #
    #     #
    #     for i in range(add_center.shape[0]):
    #         ignore_mask = np.logical_not(np.logical_and((add_center[i,0]-3.6)<pts_rect[:,0]<(add_center[i,0]+3.6),
    #                                      (add_center[i, 2] - 3.6) < pts_rect[:, 0] < (add_center[i, 2] + 3.6)))
    #         pts_rect = pts_rect[ignore_mask]
    #         pts_intensity = pts_intensity[ignore_mask]
    #
    #         pts_rect = np.concatenate((pts_rect,add_gts[i]['aug_gt_input'][:,:3]),axis=0)
    #         pts_intensity = np.concatenate((pts_intensity, add_gts[i]['aug_gt_input'][:,3].reshape(-1,1)), axis=0)
    #         gt_centers = np.concatenate((gt_centers, add_center[i]), axis=0)
    #     return pts_rect, pts_intensity, gt_centers

    def data_augmentation(self, aug_pts_rect, aug_gt_boxes3d, mustaug=False, stage=1):
        """
        :param aug_pts_rect: (N, 3)
        :param aug_gt_boxes3d: (N, 7)
        :param gt_alpha: (N)
        :return:
        """
        aug_list = cfg.AUG_METHOD_LIST
        aug_enable = 1 - np.random.rand(3)
        if mustaug is True:
            aug_enable[0] = -1
            aug_enable[1] = -1
        aug_method = []
        if 'rotation' in aug_list and aug_enable[0] < cfg.AUG_METHOD_PROB[0]:
            angle = np.random.uniform(-np.pi / cfg.AUG_ROT_RANGE, np.pi / cfg.AUG_ROT_RANGE)
            aug_pts_rect = kitti_utils.rotate_pc_along_y(aug_pts_rect, rot_angle=angle)
            aug_gt_boxes3d = kitti_utils.rotate_pc_along_y(aug_gt_boxes3d, rot_angle=angle)
            aug_method.append(['rotation', angle])

        if 'scaling' in aug_list and aug_enable[1] < cfg.AUG_METHOD_PROB[1]:
            scale = np.random.uniform(0.95, 1.05)
            aug_pts_rect = aug_pts_rect * scale
            aug_gt_boxes3d[:, 0:6] = aug_gt_boxes3d[:, 0:6] * scale
            aug_method.append(['scaling', scale])

        if 'flip' in aug_list and aug_enable[2] < cfg.AUG_METHOD_PROB[2]:
            # flip horizontal
            aug_pts_rect[:, 0] = -aug_pts_rect[:, 0]
            aug_gt_boxes3d[:, 0] = -aug_gt_boxes3d[:, 0]
            # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
            aug_method.append('flip')

        return aug_pts_rect, aug_gt_boxes3d, aug_method

    def aug_gt_dict(self, new_gt_dict):
        for gt in new_gt_dict:
            aug_points, aug_box, aug_method = self.data_augmentation(gt['points'].reshape(-1,3),gt['gt_box3d'].reshape(-1,7))
            gt['points'] = aug_points
            gt['gt_box3d'] = aug_box.reshape(-1)
            gt['obj'].pos[0] = gt['gt_box3d'][0]
            gt['obj'].pos[2] = gt['gt_box3d'][2]
        return new_gt_dict

    def apply_gt_aug_to_one_scene(self, sample_id, pts_rect, pts_intensity, all_gt_boxes3d):
        """
        :param pts_rect: (N, 3)
        :param all_gt_boxex3d: (M2, 7)
        :return:
        """
        assert self.gt_database is not None
        # extra_gt_num = np.random.randint(10, 15)
        # try_times = 50

        gt_centers = all_gt_boxes3d[:,0:3].copy()

        #generate boxes and center
        assert AUG_NUM % 3 == 0
        new_gt_dict = copy.deepcopy(random.sample(self.gt_database[1], int(AUG_NUM/3)))
        new_gt_dict += copy.deepcopy(random.sample(self.gt_database[0], int(AUG_NUM/3*2)))
        new_gt_dict = self.aug_gt_dict(new_gt_dict)
        #5 easy 5 mimic hard 5 real hard
        for i in range(int(AUG_NUM/3*2),int(AUG_NUM)): new_gt_dict[i]['presampling_flag'] = False
        add_center_ceta = np.random.uniform(0.25 * np.pi, 0.75 * np.pi, (AUG_NUM))
        add_center_depth = np.concatenate((np.random.uniform(35., 70., int(AUG_NUM/3*2)), np.random.uniform(3, 35., int(AUG_NUM/3))))
        add_center = np.zeros((AUG_NUM, 3))
        add_center[:, 0] = np.cos(add_center_ceta) * add_center_depth
        add_center[:, 2] = np.sin(add_center_ceta) * add_center_depth

        # collied detect
        gt_aug_centers = np.concatenate((gt_centers, add_center), axis=0)
        distance_gt_matrix = distance_2_numpy(gt_aug_centers[:, [0, 2]], add_center[:, [0, 2]])
        keep_id = []
        ori_gt_num = gt_centers.shape[0]
        for i in range(AUG_NUM):
            if np.min(distance_gt_matrix[i, :(i + ori_gt_num)]) > GT_DATABASE_SPARSE_DISTANCE:
                keep_id.append(i)

        new_gt_dict = [new_gt_dict[i] for i in keep_id]
        add_center = add_center[keep_id]

        # mimic hards
        for i in range(len(new_gt_dict)):
            if new_gt_dict[i]['presampling_flag']==False: continue
            sampled_mask = new_gt_dict[i]['sampled_mask']
            new_gt_dict[i]['points'] = new_gt_dict[i]['points'][sampled_mask]
            new_gt_dict[i]['intensity'] = new_gt_dict[i]['intensity'][sampled_mask]
            aug_gt_point_torch = torch.from_numpy(new_gt_dict[i]['points']).cuda().contiguous().view(1, -1, 3)
            sampled_flag_torch = pointnet2_utils.furthest_point_sample(aug_gt_point_torch,
                                                                       100)
            sampled_index = sampled_flag_torch.cpu().numpy().reshape(-1)
            new_gt_dict[i]['points'] = new_gt_dict[i]['points'][sampled_index]
            new_gt_dict[i]['intensity'] = new_gt_dict[i]['intensity'][sampled_index]

            #cutting
            # cutting_flag = np.random.uniform(0,1,1)
            # if cutting_flag>0.75:
            #     cut_sampled_index = new_gt_dict[i]['points'][:,2] > 0
            # elif cutting_flag>0.5:
            #     cut_sampled_index = new_gt_dict[i]['points'][:, 2] < 0
            # else:
            #     cut_sampled_index = new_gt_dict[i]['points'][:, 2] > -100
            # new_gt_dict[i]['points'] = new_gt_dict[i]['points'][cut_sampled_index]
            # new_gt_dict[i]['intensity'] = new_gt_dict[i]['intensity'][cut_sampled_index]

            #cutting H
            # cutting_flag = np.random.uniform(0, 1, 1)
            # if cutting_flag > 0.75:
            #     cut_sampled_index = new_gt_dict[i]['points'][:, 0] > 0.8
            # elif cutting_flag > 0.5:
            #     cut_sampled_index = new_gt_dict[i]['points'][:, 0] < 0.8
            # else:
            #     cut_sampled_index = new_gt_dict[i]['points'][:, 0] > -100
            # new_gt_dict[i]['points'] = new_gt_dict[i]['points'][cut_sampled_index]
            # new_gt_dict[i]['intensity'] = new_gt_dict[i]['intensity'][cut_sampled_index]

        # todo now is a square collied clear
        extra_gt_boxes3d = np.zeros((0,7))
        extra_gt_obj_list = []
        add_center_pts_distance_matrix = distance_2_numpy(add_center[:,[0,2]],pts_rect[:,[0,2]])
        ignore_mask = np.min(add_center_pts_distance_matrix,axis=-1)>3.6
        pts_rect = pts_rect[ignore_mask]
        pts_intensity = pts_intensity[ignore_mask]
        for i in range(add_center.shape[0]):

            # vertical noise adding
            # vert_flag = np.random.uniform(0,1,1)
            # if vert_flag>0.7:
            #     new_gt_dict[i]['points'][:, 1] += np.random.normal(0,0.1,1)

            # multi_scale insert
            # scale_flag = np.random.uniform(0,1,1)
            # if scale_flag>0.7:
            #     drange = np.random.uniform(1,4,1)
            #     scale_index = (np.abs(new_gt_dict[i]['points'][:, 0])<drange) & (np.abs(new_gt_dict[i]['points'][:, 2])<drange)
            #     new_gt_dict[i]['points'] = new_gt_dict[i]['points'][scale_index]
            #     new_gt_dict[i]['intensity'] = new_gt_dict[i]['intensity'].reshape(-1, 1)[scale_index]

            new_gt_dict[i]['points'][:, 0] += add_center[i, 0]
            new_gt_dict[i]['points'][:, 2] += add_center[i, 2]
            new_gt_dict[i]['gt_box3d'][0] = add_center[i, 0]
            new_gt_dict[i]['gt_box3d'][2] = add_center[i, 2]
            new_gt_dict[i]['obj'].pos[0] = add_center[i, 0]
            new_gt_dict[i]['obj'].pos[2] = add_center[i, 2]

            pts_rect = np.concatenate((pts_rect, new_gt_dict[i]['points']), axis=0)
            pts_intensity = np.concatenate((pts_intensity, new_gt_dict[i]['intensity'].reshape(-1, 1)), axis=0)
            extra_gt_boxes3d = np.concatenate((extra_gt_boxes3d, new_gt_dict[i]['gt_box3d'].reshape(-1, 7)), axis=0)
            extra_gt_obj_list.append(new_gt_dict[i]['obj'])
        return True, pts_rect, pts_intensity, extra_gt_boxes3d, extra_gt_obj_list



    def __len__(self):
        if cfg.RPN.ENABLED:
            return len(self.sample_id_list)
        elif cfg.RCNN.ENABLED:
            if self.mode == 'TRAIN':
                return len(self.sample_id_list)
            else:
                return len(self.image_idx_list)
        else:
            raise NotImplementedError

    def __getitem__(self, index):

        return self.get_rpn_sample(index)


    def get_rpn_sample(self, index):

        #sample data loading
        sample_id = int(self.sample_id_list[index])
        calib = self.get_calib(sample_id)
        # img = self.get_image(sample_id)
        img_shape = self.get_image_shape(sample_id)
        pts_lidar = self.get_lidar(sample_id)
        pts_lidar = pts_lidar[np.argsort(-pts_lidar[:, 2]), :]
        # get valid point (projected points should be in image)
        pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
        pts_intensity = pts_lidar[:, 3]

        #scene augmentation
        if cfg.GT_AUG_ENABLED and self.mode == 'TRAIN':
            # all labels for checking overlapping
            all_gt_obj_list = self.filtrate_objects(self.get_noise_label(sample_id))
            all_gt_boxes3d = kitti_utils.objs_to_boxes3d(all_gt_obj_list)

            gt_aug_flag = False
            if np.random.rand() < cfg.GT_AUG_APPLY_PROB:
                # augment one scene
                gt_aug_flag, pts_rect, pts_intensity, extra_gt_boxes3d, extra_gt_obj_list = \
                    self.apply_gt_aug_to_one_scene(sample_id, pts_rect, pts_intensity, all_gt_boxes3d)

        #get depth and valid points
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)
        pts_rect = pts_rect[pts_valid_flag][:, 0:3]
        pts_intensity = pts_intensity[pts_valid_flag]
        pts_depth = pts_rect_depth[pts_valid_flag]

        # generate inputs
        if self.mode == 'TRAIN' or self.random_select:
            if self.npoints < len(pts_rect):
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice), replace=False)

                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(pts_rect), dtype=np.int32)
                extra_choice = np.arange(0, len(pts_rect), dtype=np.int32)
                while self.npoints > len(choice):
                    choice = np.concatenate((choice,extra_choice),axis=0)
                choice = np.random.choice(choice, self.npoints, replace=False)
                #choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            ret_pts_rect = pts_rect[choice, :]
            ret_pts_intensity = pts_intensity[choice] - 0.5  # translate intensity to [-0.5, 0.5]
        else:
            ret_pts_rect = pts_rect
            ret_pts_intensity = pts_intensity - 0.5


        pts_features = [ret_pts_intensity.reshape(-1, 1)]
        ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]
        pts_input = np.concatenate((ret_pts_rect, ret_pts_features), axis=1)  # (N, C)


        #return if test
        if self.mode == 'TEST':
            sample_info = {'sample_id': sample_id,
                           'random_select': self.random_select,
                           'pts_input': pts_input,
                           }
            return sample_info

        #reload labels here
        noise_gt_obj_list = self.filtrate_objects(self.get_noise_label(sample_id))
        if cfg.GT_AUG_ENABLED and self.mode == 'TRAIN' and gt_aug_flag:
            noise_gt_obj_list.extend(extra_gt_obj_list)
        noise_gt_boxes3d = kitti_utils.objs_to_boxes3d(noise_gt_obj_list)

        # data augmentation
        aug_pts_input = pts_input.copy()
        aug_gt_boxes3d = noise_gt_boxes3d.copy()
        if cfg.AUG_DATA and self.mode == 'TRAIN':
            aug_pts_rect, aug_gt_boxes3d, aug_method = self.data_augmentation(aug_pts_input[:,:3], aug_gt_boxes3d)
            aug_pts_input[:,:3] = aug_pts_rect


        # generate weakly mask
        if self.mode == 'TRAIN':
            if cfg.RPN.FIXED:
                sample_info = {'sample_id': sample_id,
                               'random_select': self.random_select,
                               'pts_input': aug_pts_input,
                               'gt_centers': aug_gt_boxes3d[:, :7],
                               'aug_method': aug_method
                               }
            else:
                rpn_cls_label, rpn_reg_label = self.generate_gaussian_training_labels(aug_pts_input[:,:3], aug_gt_boxes3d[:,:3])
                # return dictionary
                sample_info = {'sample_id': sample_id,
                               'random_select': self.random_select,
                               'pts_input': aug_pts_input,
                               'rpn_cls_label': rpn_cls_label,
                               'rpn_reg_label': rpn_reg_label,
                               'gt_centers': aug_gt_boxes3d[:,:3],
                               'aug_method': aug_method
                               }

        else:
            gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
            gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
            rpn_cls_label, rpn_reg_label = self.generate_rpn_training_labels(aug_pts_input[:,:3], aug_gt_boxes3d)
            # return dictionary
            sample_info = {'sample_id': sample_id,
                           'random_select': self.random_select,
                           'pts_input': aug_pts_input,
                           'rpn_cls_label': rpn_cls_label,
                           'rpn_reg_label': rpn_reg_label,
                           'gt_boxes3d': gt_boxes3d,
                           'gt_centers': aug_gt_boxes3d[:,:3],
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

    @staticmethod
    def generate_gaussian_training_labels(pts_rect, gt_boxes3d):
        point_center_dist = np.ones((pts_rect.shape[0]), dtype=np.float32)*100
        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.float32)
        reg_label = np.zeros((pts_rect.shape[0], 3), dtype=np.float32)  # dx, dy, dz, ry, h, w, l
        dist_points2box = np.zeros((pts_rect.shape[0], gt_boxes3d.shape[0]), dtype=np.float32)
        # gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, rotate=True)
        # extend_gt_boxes3d = kitti_utils.enlarge_box3d(gt_boxes3d, extra_width=0.2)
        # extend_gt_corners = kitti_utils.boxes3d_to_corners3d(extend_gt_boxes3d, rotate=True)

        if gt_boxes3d.shape[0]>0:
            for k in range(gt_boxes3d.shape[0]):

                #class_gaussian_label
                cur_pts_rect = np.copy(pts_rect)
                #todo determined gaussian box center
                box_distance = np.sqrt(
                    np.power(cur_pts_rect[:, 0] - gt_boxes3d[k][0], 2)
                    + (np.power(cur_pts_rect[:, 1] * cfg.RPN.GAUSS_HEIGHT, 2))      # * 0.707 # gaussian height
                    + np.power(cur_pts_rect[:, 2] - gt_boxes3d[k][2], 2))
                # add_define_foreground
                point_center_dist = np.minimum(point_center_dist, np.clip(box_distance-cfg.RPN.GAUSS_STATUS,0,100)) # gaussian statics
                # box_gaussian_plus = multivariate_normal.pdf(box_distance_plus, mean=0, cov=1)
                # cls_label = np.maximum(cls_label,box_gaussian_plus)

                #box_centers
                center3d = gt_boxes3d[k][0:3].copy()  # (x, y, z)
                center3d[1] = 0.8
                center3d_tile = np.tile(center3d.reshape(1,-1),(pts_rect.shape[0], 1))
                dist_points2box[:, k] = box_distance


            cls_label = multivariate_normal.pdf(point_center_dist, mean=0, cov=cfg.RPN.GAUSS_COV) # gaussian cov
            cls_label = cls_label / (1/(math.sqrt(2*np.pi*cfg.RPN.GAUSS_COV)))


            #dist_points2box_dist = np.sqrt(np.sum(np.power(dist_points2box, 2), axis=-1))
            foreground_big_mask = np.min(dist_points2box, axis=-1) < 4.0 #(np.ones((pts_rect.shape[0]))*4.0)
            foreground_box_target = np.argmin(dist_points2box, axis=-1)
            reg_label[foreground_big_mask, 0] = gt_boxes3d[foreground_box_target][foreground_big_mask, 0] \
                                                - cur_pts_rect[foreground_big_mask][:, 0]
            reg_label[foreground_big_mask, 2] = gt_boxes3d[foreground_box_target][foreground_big_mask, 2] \
                                                - cur_pts_rect[foreground_big_mask][:, 2]
            reg_label[foreground_big_mask, 1] = 0.0

        return cls_label, reg_label

    def generate_rpn_training_labels(self, pts_rect, gt_boxes3d):
        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
        reg_label = np.zeros((pts_rect.shape[0], 3), dtype=np.float32)  # dx, dy, dz, ry, h, w, l
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, rotate=True)
        extend_gt_boxes3d = kitti_utils.enlarge_box3d(gt_boxes3d, extra_width=0.2)
        extend_gt_corners = kitti_utils.boxes3d_to_corners3d(extend_gt_boxes3d, rotate=True)
        for k in range(gt_boxes3d.shape[0]):
            box_corners = gt_corners[k]
            fg_pt_flag = kitti_utils.in_hull(pts_rect, box_corners)
            fg_pts_rect = pts_rect[fg_pt_flag]
            cls_label[fg_pt_flag] = 1

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_gt_corners[k]
            fg_enlarge_flag = kitti_utils.in_hull(pts_rect, extend_box_corners)
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1

            # pixel offset of object center
            center3d = gt_boxes3d[k][0:3].copy()  # (x, y, z)
            center3d[1] = 0
            reg_label[fg_pt_flag, 0:3] = center3d - fg_pts_rect  # Now y is the true center of 3d box 20180928
            reg_label[:,1] = 0
        return cls_label, reg_label


    def collate_batch(self, batch):
        if self.mode != 'TRAIN' and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            assert batch.__len__() == 1
            return batch[0]

        batch_size = batch.__len__()
        ans_dict = {}

        for key in batch[0].keys():

            if cfg.RPN.ENABLED and key=='gt_centers':
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, batch[k][key].__len__())
                batch_gt_centers = np.zeros((batch_size, max_gt, 3), dtype=np.float32)
                for i in range(batch_size):
                    batch_gt_centers[i, :batch[i][key].__len__(), :] = batch[i][key]
                ans_dict[key] = batch_gt_centers
                continue

            if cfg.RPN.ENABLED and key=='gt_boxes3d':
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


if __name__ == '__main__':
    pass