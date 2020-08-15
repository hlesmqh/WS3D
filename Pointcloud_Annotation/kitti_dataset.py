import os
import numpy as np
import torch
import torch.utils.data as torch_data
import calibration as calibration
import kitti_utils as kitti_utils
from PIL import Image
import cv2




class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='train'):
        self.split = split
        self.classes = ['Car', 'Van', 'Truck']
        is_test = self.split == 'test'
        self.imageset_dir = os.path.join(root_dir, 'object', 'testing' if is_test else 'training')

        split_dir = os.path.join(root_dir, 'object', 'ImageSets', split + '.txt')
        self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.num_sample = self.image_idx_list.__len__()

        self.image_dir = os.path.join(self.imageset_dir, 'image_2')
        self.lidar_dir = os.path.join(self.imageset_dir, 'velodyne')
        self.calib_dir = os.path.join(self.imageset_dir, 'calib')
        self.label_dir = os.path.join(self.imageset_dir, 'label_2')
        self.plane_dir = os.path.join(self.imageset_dir, 'planes')

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
        valid_obj_list = []
        type_whitelist = self.classes
        for obj in obj_list:
            if obj.cls_type not in type_whitelist:
                continue
            valid_obj_list.append(obj)
        return valid_obj_list


    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        sample_id = int(self.image_idx_list[index])

        calib = self.get_calib(sample_id)
        image = self.get_image(sample_id)
        pts_lidar = self.get_lidar(sample_id)[:,:3]
        pts_lidar = pts_lidar[np.argsort(-pts_lidar[:,2]), :]
        pts_rect = calib.lidar_to_rect(pts_lidar)
        pts_image, _ = calib.rect_to_img(pts_rect)
        obj_list = self.filtrate_objects(self.get_label(sample_id))

        gt_boxes3d_vel = np.zeros((obj_list.__len__(), 7), dtype=np.float32)
        gt_boxes3d_cam = np.zeros((obj_list.__len__(), 7), dtype=np.float32)
        gt_boxes2d_cam = np.zeros((obj_list.__len__(), 4), dtype=np.float32)

        for k, obj in enumerate(obj_list):
            # True for points in boxes
            gt_boxes3d_vel[k, 0:3], gt_boxes3d_vel[k, 3], gt_boxes3d_vel[k, 4], gt_boxes3d_vel[k, 5], \
            gt_boxes3d_vel[k, 6] \
                = calib.box_center_camera_to_lidar(obj.pos), obj.h, obj.w, obj.l, obj.alpha  # obj.ry

            gt_boxes3d_cam[k, 0:3], gt_boxes3d_cam[k, 3], gt_boxes3d_cam[k, 4], gt_boxes3d_cam[k, 5], \
            gt_boxes3d_cam[k, 6] \
                = obj.pos, obj.h, obj.w, obj.l, obj.ry  # obj.ry

            gt_boxes2d_cam[k, 0:4] = obj.box2d

        pt_mask_flag = np.zeros((len(obj_list), np.shape(pts_rect)[0]))

        # gt_foreground
        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d_cam, rotate=True)
        for k in range(gt_boxes3d_cam.shape[0]):
            fg_pt_flag = kitti_utils.in_hull(pts_rect, gt_corners[k])
            cls_label[fg_pt_flag] = 1

        valid_region = np.logical_and(pts_lidar[:, 0]>np.abs(pts_lidar[:, 1]), pts_lidar[:, 0]>0)


        data ={
            'sample_id': sample_id,
            'image': image,
            'calib': calib,
            'pts_lidar': pts_lidar[valid_region],
            'pts_rect': pts_rect[valid_region],
            'pts_image': pts_image[valid_region],
            'gt_boxes_3d_vel': gt_boxes3d_vel,
            'gt_boxes_3d_cam': gt_boxes3d_cam,
            'pt_mask_flag': cls_label,
            'gt_boxes2d_cam': gt_boxes2d_cam,
        }


        return data
