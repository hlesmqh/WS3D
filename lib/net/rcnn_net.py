import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetSAModule
from lib.utils.bbox_transform import decode_center_target, decode_bbox_target_stage_2, center_box2box, box2center_box, refine_box
import pointnet2_lib.pointnet2.pytorch_utils as pt_utils
import lib.utils.loss_utils as loss_utils
from lib.config import cfg
from lib.net.transformer import Transformer
import numpy as np
import matplotlib.pyplot as plt
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils


class RCNNNet(nn.Module):
    def __init__(self, num_classes, num_point=512, input_channels=0, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        self.ATT_modules = nn.ModuleList()
        channel_in = input_channels
        self.MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()


        #todo use statics feature num
        #self.rcnn_input_channel = 3 + int(cfg.RCNN.USE_INTENSITY) + int(cfg.RCNN.USE_MASK) + int(cfg.RCNN.USE_DEPTH)
        self.rcnn_input_channel = 5
        self.input_tansformer = Transformer(num_point, 3)
        self.xyz_up_layer = pt_utils.SharedMLP([3] + cfg.RCNN.XYZ_UP_LAYER,
                                               bn=cfg.RCNN.USE_BN)

        #self.feature_tansformer = Transformer(num_point, cfg.RCNN.XYZ_UP_LAYER[-1])

        self.feature_up_layer = pt_utils.SharedMLP([self.rcnn_input_channel-3] + cfg.RCNN.XYZ_UP_LAYER,
                                               bn=cfg.RCNN.USE_BN)
        c_out = cfg.RCNN.XYZ_UP_LAYER[-1]
        self.merge_down_layer = pt_utils.SharedMLP([c_out * 2, c_out], bn=cfg.RCNN.USE_BN)

        for k in range(cfg.RCNN.SA_CONFIG.NPOINTS.__len__()):

            if cfg.ATTENTION:
                self.ATT_modules.append(pt_utils.SharedMLP([channel_in], bn=cfg.RCNN.USE_BN, activation=nn.ReLU(inplace=True)))

            mlps = [channel_in] + cfg.RCNN.SA_CONFIG.MLPS[k]

            npoint = cfg.RCNN.SA_CONFIG.NPOINTS[k] if cfg.RCNN.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=npoint,
                    radius=cfg.RCNN.SA_CONFIG.RADIUS[k],
                    nsample=cfg.RCNN.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=use_xyz,
                    bn=cfg.RCNN.USE_BN
                )
            )
            channel_in = mlps[-1]

            # class SharedMLP(nn.Sequential):
            #
            #     def __init__(
            #             self,
            #             args: List[int],
            #             *,
            #             bn: bool = False,
            #             activation=nn.ReLU(inplace=True),
            #             preact: bool = False,
            #             first: bool = False,
            #             name: str = "",
            #             instance_norm: bool = False,
            #     ):


        # classification layer
        cls_channel = 1 if num_classes == 2 else num_classes
        cls_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RCNN.FOCAL_ALPHA[0],
                                                                           gamma=cfg.RCNN.FOCAL_GAMMA)
        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            self.cls_loss_func = F.binary_cross_entropy
        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy':
            cls_weight = torch.from_numpy(cfg.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduce=False, weight=cls_weight)
        else:
            raise NotImplementedError

        # regression layer
        per_loc_bin_num = int(cfg.RCNN.LOC_SCOPE / cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(cfg.RCNN.LOC_Y_SCOPE / cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        reg_channel += (1 if not cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2)

        reg_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)

        # IOU estimation
        # IOU layer
        if cfg.IOUN.ENABLED:
            self.cascade = cfg.CASCADE
            self.can_xyz_up_layer = nn.ModuleList()
            self.can_feature_up_layer = nn.ModuleList()
            self.can_merge_down_layer = nn.ModuleList()
            self.SA_score_modules = nn.ModuleList()
            self.ATT_score_modules = nn.ModuleList()
            self.IOU_layer = nn.ModuleList()
            self.ICL_layer = nn.ModuleList()
            self.ref_layer = nn.ModuleList()
            for i in range(self.cascade):
                for p in self.parameters():
                    p.requires_grad = False



                self.can_xyz_up_layer.append(pt_utils.SharedMLP([3] + cfg.RCNN.XYZ_UP_LAYER,
                                                              bn=cfg.RCNN.USE_BN).cuda())
                self.can_feature_up_layer.append(pt_utils.SharedMLP([2] + cfg.RCNN.XYZ_UP_LAYER,
                                                                bn=cfg.RCNN.USE_BN).cuda())
                c_out = cfg.RCNN.XYZ_UP_LAYER[-1]
                self.can_merge_down_layer.append(pt_utils.SharedMLP([c_out * 2, c_out], bn=cfg.RCNN.USE_BN))

                iou_channel_in = input_channels
                for k in range(cfg.IOUN.SA_CONFIG.NPOINTS.__len__()):


                    mlps = [iou_channel_in] + cfg.IOUN.SA_CONFIG.MLPS[k]

                    if cfg.ATTENTION:
                        self.ATT_score_modules.append(pt_utils.SharedMLP([iou_channel_in], bn=cfg.RCNN.USE_BN, activation=nn.ELU(inplace=True)))

                    npoint = cfg.IOUN.SA_CONFIG.NPOINTS[k] if cfg.IOUN.SA_CONFIG.NPOINTS[k] != -1 else None
                    self.SA_score_modules.append(
                        PointnetSAModule(
                            npoint=npoint,
                            radius=cfg.IOUN.SA_CONFIG.RADIUS[k],
                            nsample=cfg.IOUN.SA_CONFIG.NSAMPLE[k],
                            mlp=mlps,
                            use_xyz=use_xyz,
                            bn=cfg.IOUN.USE_BN
                        ).cuda()
                    )
                    iou_channel_in = mlps[-1]


                IOU_channel = 1
                IOU_layers = []
                pre_channel = iou_channel_in
                for k in range(0, cfg.IOUN.CLS_FC.__len__()):
                    IOU_layers.append(pt_utils.Conv1d(pre_channel, cfg.IOUN.CLS_FC[k], bn=cfg.IOUN.USE_BN))
                    pre_channel = cfg.IOUN.CLS_FC[k]
                IOU_layers.append(pt_utils.Conv1d(pre_channel, IOU_channel, activation=None))
                if cfg.IOUN.DP_RATIO >= 0:
                    IOU_layers.insert(1, nn.Dropout(cfg.IOUN.DP_RATIO))
                self.IOU_layer.append(nn.Sequential(*IOU_layers).cuda())

                ICL_channel = 1
                ICL_layers = []
                pre_channel = iou_channel_in
                for k in range(0, cfg.IOUN.CLS_FC.__len__()):
                    ICL_layers.append(pt_utils.Conv1d(pre_channel, cfg.IOUN.CLS_FC[k], bn=cfg.IOUN.USE_BN))
                    pre_channel = cfg.IOUN.CLS_FC[k]
                ICL_layers.append(pt_utils.Conv1d(pre_channel, ICL_channel, activation=None))
                if cfg.IOUN.DP_RATIO >= 0:
                    ICL_layers.insert(1, nn.Dropout(cfg.IOUN.DP_RATIO))
                self.ICL_layer.append(nn.Sequential(*ICL_layers).cuda())

                per_loc_bin_num = int(cfg.IOUN.LOC_SCOPE / cfg.IOUN.LOC_BIN_SIZE) * 2
                loc_y_bin_num = int(cfg.IOUN.LOC_Y_SCOPE / cfg.IOUN.LOC_Y_BIN_SIZE) * 2
                ref_channel = 7

                ref_layers = []
                pre_channel = iou_channel_in
                for k in range(0, cfg.IOUN.REG_FC.__len__()):
                    ref_layers.append(pt_utils.Conv1d(pre_channel, cfg.IOUN.REG_FC[k], bn=cfg.IOUN.USE_BN))
                    pre_channel = cfg.IOUN.REG_FC[k]
                ref_layers.append(pt_utils.Conv1d(pre_channel, ref_channel, activation=None))
                if cfg.IOUN.DP_RATIO >= 0:
                    ref_layers.insert(1, nn.Dropout(cfg.IOUN.DP_RATIO))
                self.ref_layer.append(nn.Sequential(*ref_layers).cuda())

        self.init_weights(weight_init='xavier')




    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def get_rotation_matrix(self, batch_size, ry):
        Rot_y = torch.zeros((batch_size, 3, 3)).cuda()
        Rot_y[:, 0, 0] = torch.cos(ry[:, 0])
        Rot_y[:, 0, 2] = torch.sin(ry[:, 0])
        Rot_y[:, 1, 1] = 1
        Rot_y[:, 2, 0] = -torch.sin(ry[:, 0])
        Rot_y[:, 2, 2] = torch.cos(ry[:, 0])
        return Rot_y

    def forward(self, input_data):
        """
        :param input_data: input dict
        :return:
        """
        if cfg.RCNN.ROI_SAMPLE_JIT:
            xyz, raw_features = input_data['cur_box_point'], torch.cat((input_data['cur_box_reflect'], input_data['train_mask']), dim=-1)
            features = torch.cat((xyz, raw_features), dim=-1)

        if cfg.RCNN.USE_RPN_FEATURES:

            # xyz_input = pts_input[..., 0:self.rcnn_input_channel].transpose(1, 2).unsqueeze(dim=3)
            xyz_input = xyz.transpose(1, 2).unsqueeze(dim=3)
            raw_features_input = raw_features.transpose(1, 2).unsqueeze(dim=3)

            uper_xyz = self.xyz_up_layer(xyz_input)
            uper_feature = self.feature_up_layer(raw_features_input)

            #use rpn feature
            if 'cur_pts_feature' in input_data.keys():
                uper_feature = input_data['cur_pts_feature'].transpose(1, 2).unsqueeze(dim=3)

            merged_feature = torch.cat((uper_xyz, uper_feature), dim=1)
            merged_feature = self.merge_down_layer(merged_feature)


            l_xyz, l_features = [xyz], [merged_feature.squeeze(dim=3)]
        else:
            l_xyz, l_features = [xyz], [features]




        for i in range(len(self.SA_modules)):

            if cfg.ATTENTION:
                mean_channels = np.sqrt(float(l_features[i].shape[1]))
                context = self.ATT_modules[i](l_features[i].unsqueeze(dim=3)).squeeze(dim=3)
                attention = F.softmax(torch.bmm(l_features[i].transpose(1,2),l_features[i])/mean_channels,dim=1)
                l_features[i] = torch.bmm(
                    context,
                    attention
                    )+l_features[i]

            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        rcnn_cls = self.cls_layer(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layer(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        #predict box
        roi_boxes3d = torch.zeros((rcnn_reg.shape[0], 3)).cuda()
        pred_boxes3d_ce = decode_bbox_target_stage_2(roi_boxes3d.view(-1, 3), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                                     anchor_size=self.MEAN_SIZE,
                                                     loc_scope=cfg.RCNN.LOC_SCOPE,
                                                     loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                                     num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                                     get_xz_fine=False,
                                                     loc_y_scope=cfg.RCNN.LOC_Y_SCOPE,
                                                     loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                                     get_ry_fine=False).view(-1, 1, 7).detach()

        #center regression
        #pred_boxes3d = center_box2box(pred_boxes3d_ce)
        #ground regression
        pred_boxes3d = pred_boxes3d_ce.clone()
        pred_boxes3d_ce = box2center_box(pred_boxes3d_ce)

        ret_dict = {'rcnn_cls': rcnn_cls,
                    'rcnn_reg': rcnn_reg,
                    'pred_boxes3d':pred_boxes3d} #, 'T1':T1, 'T2':T2}

        if cfg.IOUN.ENABLED:

            SA_SCORE_layer=0
            for c in range(self.cascade):
                xyz = input_data['cur_box_point'].clone()
                if c !=0:
                    rcnn_ref = rcnn_ref.view(rcnn_ref.shape[0], 1, rcnn_ref.shape[-1])
                    pred_boxes3d_ce = refine_box(pred_boxes3d_ce.view(-1,7), rcnn_ref.view(rcnn_ref.shape[0], rcnn_ref.shape[-1]))
                    pred_boxes3d_ce = pred_boxes3d_ce.view(-1,1,7)

                #noise the prediction
                if 'iou_trans' in input_data.keys():
                    iou_trans_noise = input_data['iou_trans'][...,c]
                    iou_scale_noise = input_data['iou_scale'][...,c]
                    iou_ry_noise = input_data['iou_ry'][...,c]

                    # location
                    pred_boxes3d_ce[:, :, 0:3] += iou_trans_noise
                    # size
                    pred_boxes3d_ce[:, :, 3:6] *= iou_scale_noise
                    # angle
                    pred_boxes3d_ce[:, :, 6] += iou_ry_noise[:,:,0]

                #canional the points by box
                xyz[:, :, 0] = xyz[:, :, 0] - pred_boxes3d_ce[:, :, 0]
                xyz[:, :, 1] = xyz[:, :, 1] - pred_boxes3d_ce[:, :, 1]
                xyz[:, :, 2] = xyz[:, :, 2] - pred_boxes3d_ce[:, :, 2]

                Rot_y = self.get_rotation_matrix(xyz.shape[0], -pred_boxes3d_ce[:, :, 6])
                canional_xyz = torch.einsum('ijk,ikl->ijl', xyz, Rot_y.permute(0, 2, 1))

                # norm transfer
                extend_factor = 1.2
                canional_xyz[:, :, 0] = canional_xyz[:, :, 0] / (pred_boxes3d_ce[:, :, 5] / 2)
                canional_xyz[:, :, 1] = canional_xyz[:, :, 1] / (pred_boxes3d_ce[:, :, 3] / 2)
                canional_xyz[:, :, 2] = canional_xyz[:, :, 2] / (pred_boxes3d_ce[:, :, 4] / 2)
                canional_mask = torch.max(torch.abs(canional_xyz), dim=-1)[0] > extend_factor
                canional_xyz[canional_mask] = canional_xyz[canional_mask] = 0.0


                #network
                can_xyz_input = canional_xyz.transpose(1, 2).unsqueeze(dim=3)
                uper_can_xyz = self.can_xyz_up_layer[c](can_xyz_input)
                uper_can_feature = self.can_feature_up_layer[c](raw_features_input)
                can_merged_feature = torch.cat((uper_can_xyz, uper_can_feature), dim=1)
                can_merged_feature = self.can_merge_down_layer[c](can_merged_feature)


                #no merge
                # can_merged_feature = uper_can_xyz

                l_xyz, l_features = [canional_xyz], [can_merged_feature.squeeze(dim=3)]

                for i in range(int(len(self.SA_score_modules)/cfg.CASCADE)):

                    if cfg.ATTENTION:
                        mean_channels = np.sqrt(float(l_features[i].shape[1]))
                        context = self.ATT_score_modules[SA_SCORE_layer+i](l_features[i].unsqueeze(dim=3)).squeeze(dim=3)
                        attention = F.softmax(torch.bmm(l_features[i].transpose(1,2),l_features[i])/mean_channels,dim=1)
                        l_features[i] = torch.bmm(
                            context,
                            attention
                            )+l_features[i]

                    li_xyz, li_features = self.SA_score_modules[SA_SCORE_layer+i](l_xyz[i], l_features[i])
                    l_xyz.append(li_xyz)
                    l_features.append(li_features)
                SA_SCORE_layer +=i+1

                rcnn_iou = self.IOU_layer[c](l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
                rcnn_ref = self.ref_layer[c](l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
                ioun_cls = self.ICL_layer[c](l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)

                pred_boxes3d = center_box2box(pred_boxes3d_ce).view(-1,1,7)

                refined_box = refine_box(pred_boxes3d.view(-1, 7),
                                             rcnn_ref.view(rcnn_ref.shape[0], rcnn_ref.shape[-1])).view(-1,1,7)

                ret_dict.update({'rcnn_iou': rcnn_iou,
                                 'rcnn_ref': rcnn_ref,
                                 'ioun_cls': ioun_cls,
                                 'pred_boxes3d':pred_boxes3d,
                                 'refined_box':refined_box})

        ret_dict.update(input_data)
        return ret_dict
