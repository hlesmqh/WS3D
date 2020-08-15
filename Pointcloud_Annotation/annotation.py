import matplotlib.pyplot as plt
from colour import Color
import numpy as np
from kitti_dataset import KittiDataset
from torch.utils.data import DataLoader
import time
from scipy.stats import multivariate_normal
import matplotlib
import matplotlib.patches as patches
print(matplotlib.use('Qt5Agg'))







dataset = KittiDataset('/media/qinghaomeng/DATA/Kitti', split='train')
test_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True,
                         num_workers=1)

rect1 = [0.05,0.6,0.9,0.35]
rect2 = [0.09,0.05,0.32,0.48]
rect3 = [0.59,0.05,0.32,0.48]



class Exchange():
    def __init__(self, sample_id, pts_lidar, pts_rect, pts_image, image, fig, ax1_img, ax2_rvel, ax3_gvel, gt_boxes2d_cam):

        self.image_center = []
        self.bev_center = []
        self.sample_id = sample_id
        self.pts_lidar = pts_lidar
        self.pts_rect = pts_rect
        self.pts_image = pts_image
        self.image = image
        self.center_heatmap = np.zeros((np.shape(pts_lidar)[0]))
        self.bev_region_center = [5, 0, 0]
        self.fig = fig
        self.ax1_img = ax1_img
        self.ax2_rvel = ax2_rvel
        self.ax3_gvel = ax3_gvel
        self.gt_boxes2d_cam = gt_boxes2d_cam





        self.print_front_lidar()
        self.print_region_bev()
        self.print_global_bev()


        #draw plot
        # TODO front lidar
    def print_front_lidar(self):
        self.ax1_img.set_title('%06d/%d'%(self.sample_id, len(open('label_w/label.txt', 'r').readlines())))

        self.ax1_img.scatter(self.pts_image[:,0], self.pts_image[:,1], s=10, c=self.center_heatmap, edgecolor='none', cmap=plt.get_cmap('rainbow'), alpha=1, marker='.', vmin=0, vmax=np.max(self.center_heatmap))
        #if self.gt_boxes2d_cam.shape[0]>0:
#        for b in range(self.gt_boxes2d_cam.shape[0]):
#            self.ax1_img.add_patch(patches.Rectangle((gt_boxes2d_cam[b,0], gt_boxes2d_cam[b,1]),
#                        gt_boxes2d_cam[b,2]-gt_boxes2d_cam[b,0], gt_boxes2d_cam[b,3]-gt_boxes2d_cam#[b,1], linewidth=1, edgecolor='white',
#                        facecolor='none'))
        self.ax1_img.imshow(self.image)
        self.ax1_img.axis('off')



        # TODO region bev vieew
    def print_region_bev(self):
        x, z = self.bev_region_center[0], self.bev_region_center[2]
        cur_pts_rect = np.copy(self.pts_rect)
        X_axis = np.clip(cur_pts_rect[:, 0] - x, -4, 4)
        x_region_point = np.logical_and(X_axis < 4, X_axis > -4)
        Z_axis = np.clip(cur_pts_rect[:, 2] - z, -4, 4)
        z_region_point = np.logical_and(Z_axis < 4, Z_axis > -4)
        region_point = np.logical_and(x_region_point, z_region_point)
        X_axis = X_axis[region_point]
        Z_axis = Z_axis[region_point]
        Y_axis = np.clip(cur_pts_rect[:, 1], -4, 4)[region_point]
        self.ax2_rvel.grid(True)
        self.ax2_rvel.axis([-4, 4, -4, 4])
        self.ax2_rvel.scatter(X_axis, Z_axis, s=20, c=Y_axis, edgecolor='none', cmap=plt.get_cmap('YlOrRd'), alpha=1,
                         marker='.',vmin=-2,vmax=3)

        # TODO global bev vieew
    def print_global_bev(self):
        ax3_gvel.axis([-35, 35, 0, 60])
        X_axis = np.clip(self.pts_lidar[:, 0], 0, 60)
        Y_axis = -np.clip(self.pts_lidar[:, 1], -35, 35)
        self.ax3_gvel.scatter(Y_axis, X_axis, s=6, c=self.center_heatmap, edgecolor='none', cmap=plt.get_cmap('rainbow'), alpha=1,
                         marker='.', vmin=0, vmax=np.max(self.center_heatmap))


    def on_key_press(self, event):
        if event.inaxes == ax1_img:
            img_x, img_y = event.xdata, event.ydata
            self.ax2_rvel.cla()
            current_pts_image = np.copy(self.pts_image)
            current_pts_image[:, 0] = current_pts_image[:, 0] - img_x
            current_pts_image[:, 1] = current_pts_image[:, 1] - img_y
            img_dist = np.sum(np.abs(current_pts_image), axis=1)
            cur_img_center_index = np.argsort(img_dist)[0]
            self.image_center.append([img_x, img_y])
            self.bev_region_center = self.pts_rect[cur_img_center_index]
            self.print_region_bev()
            self.fig.canvas.draw()

        elif event.inaxes == ax2_rvel:
            reg_vel_x, reg_vel_z = event.xdata, event.ydata
            self.bev_region_center
            vel_x, vel_z = self.bev_region_center[0] + reg_vel_x, self.bev_region_center[2] + reg_vel_z
            self.bev_center.append([vel_x,vel_z])
            cur_pts_rect = np.copy(self.pts_rect)

            box_distance = np.sqrt(np.power(cur_pts_rect[:,0] - vel_x, 2) + np.power(cur_pts_rect[:,1]-0.8, 2) + np.power(cur_pts_rect[:,2] - vel_z, 2))
            box_gaussian = multivariate_normal.pdf(box_distance,mean=0,cov=2)
            self.center_heatmap += box_gaussian

            self.ax1_img.cla()
            self.ax3_gvel.cla()
            self.print_front_lidar()
            self.print_global_bev()
            self.fig.canvas.draw()

        # elif event.inaxes == ax3_gvel:
        else:
            #plt.savefig('image_w/%06d.png'%self.sample_id, dpi=300)
            plt.close()


    #print(in_axes)

for data in dataset:
        sample_id = data['sample_id']
        endline = open('label_w/label.txt', 'r').readlines()[-1]
        if sample_id < int(endline.split(' ')[0]):
            continue
        image = data['image'][:,:,[2,1,0]]
        calib = data['calib']
        pts_lidar = data['pts_lidar']
        pts_rect = data['pts_rect']
        pts_image = data['pts_image']
        gt_boxes3d_vel = data['gt_boxes_3d_vel']
        gt_boxes3d_cam = data['gt_boxes_3d_cam']
        pt_mask_flag = data['pt_mask_flag']
        gt_boxes2d_cam = data['gt_boxes2d_cam']

        fig = plt.figure(figsize=(15, 10))

        ax1_img = plt.axes(rect1)
        ax2_rvel = plt.axes(rect2, facecolor='dimgray')
        ax3_gvel = plt.axes(rect3, facecolor='dimgray')

        exchange = Exchange(sample_id, pts_lidar, pts_rect, pts_image, image, fig, ax1_img, ax2_rvel, ax3_gvel, gt_boxes2d_cam)
        fig.canvas.mpl_connect('button_press_event', exchange.on_key_press)

        plt.show()
        if gt_boxes3d_cam.shape[0]>0:
            f = open('label_w/label.txt', 'a+')
            for center in exchange.bev_center:
                center_box_distance = np.sqrt(np.power(center[0] - gt_boxes3d_cam[:,0], 2) + np.power(center[1] - gt_boxes3d_cam[:,2], 2))
                if np.min(center_box_distance)<3.0:
                    index = np.argmin(center_box_distance)
                    f.write('%06d '%sample_id + ' '.join([str(x) for x in center]) + ' %f'%gt_boxes3d_cam[index,0] + ' %f'%gt_boxes3d_cam[index,2] +'\n')
            f.close()
        # 000599
        # 661

        #time.sleep(1)











        #TODO plot vieew
        # color = raw_pointmap[:, :, 0].reshape(-1)# *100.
        # plt.rcParams['figure.figsize'] =(15,5)
        # plt.scatter(X, Y, s=10, c=color, edgecolor='none',cmap=plt.get_cmap('rainbow'), alpha=1, marker='|')
        # plt.show()
        # color = point_cls_label[:, :].reshape(-1)  # *100.
        # plt.rcParams['figure.figsize'] = (15, 5)
        # gaussian_color = np.zeros(np.shape(point_xyz[:,0]))
        # plt.scatter(X, Y, s=10, c=color, edgecolor='none', cmap=plt.get_cmap('Blues'), alpha=1, marker='|')
        # plt.show()




        #
        #

        #
        #
        #
        #
        # img = get_image(sample_id)
        #
        # gaussian_color = np.zeros(np.shape(point_xyz[:,0]))
        # for i in range(np.shape(gt_boxes3d_vel)[0]):
        #     box = gt_boxes3d_vel[i,:]
        #     box_distance = np.sqrt(np.power(point_xyz[:,0] - box[0], 2) + np.power(point_xyz[:,1] - box[1], 2) + np.power(point_xyz[:,2] - box[2] - box[3]/2, 2))
        #     box_gaussian = multivariate_normal.pdf(box_distance,mean=0,cov=1.5)
        #     gaussian_color += box_gaussian
        #
        # # color = Z_axis #height map
        # # color = gaussian_color
        # plt.scatter(pts_image[:,0], pts_image[:,1], s=20, c=np.clip(point_xyz[:, 2], -2, 2), edgecolor='none', cmap=plt.get_cmap('viridis'), alpha=1, marker='.')
        # plt.imshow(img)
        #
        # plt.axis('off')
        # #todo Online display solution: ask server in terminal: watch -n 0 display showcls.jpg
        # # plt.savefig('showcls.jpg',dpi=150)
        # plt.show()





