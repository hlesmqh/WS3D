import os
import numpy as np
import torch.utils.data as torch_data
import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils
from PIL import Image
from copy import deepcopy
import cv2


class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='train', noise=None):
        self.split = split
        self.classes = ['Car']
        is_test = self.split == 'test'
        self.imageset_dir = os.path.join(root_dir, 'testing' if is_test else 'training')

        split_dir = os.path.join(root_dir, 'ImageSets', split + '.txt')
        self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.num_sample = self.image_idx_list.__len__()

        self.image_dir = os.path.join(self.imageset_dir, 'image_2')
        self.lidar_dir = os.path.join(self.imageset_dir, 'velodyne')
        self.calib_dir = os.path.join(self.imageset_dir, 'calib')
        self.label_dir = os.path.join(self.imageset_dir, 'label_2')
        self.plane_dir = os.path.join(self.imageset_dir, 'planes')
        self.num_class = self.classes.__len__()

        self.type_whitelist = self.classes
        if noise=='label_noise':
            self.noisy_label_dir = os.path.join(self.imageset_dir, noise)
        else:
            self.noisy_label_dir = deepcopy(self.label_dir)

        if split == 'train':
            if 'Car' in self.classes:
                self.type_whitelist.append('Van')
            if 'Pedestrian' in self.classes:  # or 'Cyclist' in self.classes:
                self.type_whitelist.append('Person_sitting')
        print(self.type_whitelist)

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file)  # (H, W, 3) BGR mode

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti_utils.get_objects_from_label(label_file)

    def get_label_noise(self, idx):
        label_file = os.path.join(self.noisy_label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti_utils.get_objects_from_label(label_file)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.plane_dir, '%06d.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def filtrate_objects(self, obj_list):
        """
        Discard objects which are not in self.classes (or its similar classes)
        :param obj_list: list
        :return: list
        """

        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in self.type_whitelist:  # rm Van, 20180928
                continue
            if self.split=='train' and (self.check_pc_range(obj.pos) is False):
                continue
            valid_obj_list.append(obj)
        return valid_obj_list

    def check_pc_range(self, xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range, y_range, z_range = [-40, 40], [-3,   3], [0, 70.4]
        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

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
        PC_REDUCE_BY_RANGE = True
        if PC_REDUCE_BY_RANGE:
            x_range, y_range, z_range = [-40, 40], [-3,   3], [0, 70.4]
            pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
            range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
            pts_valid_flag = pts_valid_flag & range_flag
        return pts_valid_flag


    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        sample_id = int(self.image_idx_list[index])

        calib = self.get_calib(sample_id)
        image = self.get_image(sample_id)
        img_shape = self.get_image_shape(sample_id)
        full_lidar= self.get_lidar(sample_id)
        pts_lidar = full_lidar[:, :3]
        pts_reflect = full_lidar[:, -1]
        pts_reflect = pts_reflect[np.argsort(-pts_lidar[:, 2])]
        pts_lidar = pts_lidar[np.argsort(-pts_lidar[:, 2]), :]
        pts_rect = calib.lidar_to_rect(pts_lidar)
        pts_image, pts_rect_depth = calib.rect_to_img(pts_rect)

        valid_region = self.get_valid_flag(pts_rect, pts_image, pts_rect_depth, img_shape)
        #valid_region = np.logical_and(pts_lidar[:, 0] > np.abs(pts_lidar[:, 1]), pts_lidar[:, 0] > 0)

        if not self.split=='test':

            obj_list = self.filtrate_objects(self.get_label(sample_id))
            noise_obj_list = self.filtrate_objects(self.get_label_noise(sample_id))

            gt_boxes3d_cam = np.zeros((obj_list.__len__(), 7), dtype=np.float32)
            gt_boxes2d_cam = np.zeros((obj_list.__len__(), 4), dtype=np.float32)
            gt_alpha = np.zeros((obj_list.__len__()), dtype=np.float32)

            for k, obj in enumerate(obj_list):

                gt_boxes3d_cam[k, 0:3], gt_boxes3d_cam[k, 3], gt_boxes3d_cam[k, 4], gt_boxes3d_cam[k, 5], \
                gt_boxes3d_cam[k, 6] \
                    = obj.pos, obj.h, obj.w, obj.l, obj.ry  # obj.ry

                gt_boxes2d_cam[k, 0:4] = obj.box2d
                gt_alpha = obj.alpha

            noise_gt_boxes3d_cam = np.zeros((noise_obj_list.__len__(), 7), dtype=np.float32)
            for k, obj in enumerate(noise_obj_list):

                noise_gt_boxes3d_cam[k, 0:3], noise_gt_boxes3d_cam[k, 3], noise_gt_boxes3d_cam[k, 4], noise_gt_boxes3d_cam[k, 5], \
                noise_gt_boxes3d_cam[k, 6] \
                    = obj.pos, obj.h, obj.w, obj.l, obj.ry  # obj.ry

        if not self.split == 'test':
            data = {
                'sample_id': sample_id,
                'image': image,
                'calib': calib,
                'pts_lidar': pts_lidar[valid_region],
                'pts_rect': pts_rect[valid_region],
                'pts_reflect': pts_reflect[valid_region] - 0.5,
                'pts_image': pts_image[valid_region],
                'gt_boxes_3d_cam': gt_boxes3d_cam,
                'gt_boxes2d_cam': gt_boxes2d_cam,
                'noise_gt_boxes3d_cam': noise_gt_boxes3d_cam,
                'gt_alpha': gt_alpha,
            }
        else:
            data = {
                'sample_id': sample_id,
                'image': image,
                'calib': calib,
                'pts_lidar': pts_lidar[valid_region],
                'pts_rect': pts_rect[valid_region],
                'pts_reflect': pts_reflect[valid_region] - 0.5,
                'pts_image': pts_image[valid_region],
            }

        return data
