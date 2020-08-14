import torch
import torch.nn as nn
from lib.net.rpn import RPN
from lib.net.rcnn_net import RCNNNet
from lib.config import cfg
#from torch_cluster import fps


class PointRCNN(nn.Module):
    def __init__(self, num_classes, num_point=512, use_xyz=True, mode='TRAIN',old_model=False):
        super().__init__()
        self.mode = mode
        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED or cfg.IOUN.ENABLED

        if cfg.RPN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode,old_model=old_model)

        if cfg.RCNN.ENABLED or cfg.IOUN.ENABLED:
            rcnn_input_channels = 128  # channels of rpn features x,y,z,r,mask
            #self.rcnn_net = RCNNNet(num_point=num_point, num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            self.rcnn_net = RCNNNet(num_point=num_point, num_classes=num_classes, input_channels=rcnn_input_channels,
                                    use_xyz=use_xyz)

    def forward(self, input_data):
        if cfg.RPN.ENABLED:
            output = {}
            # rpn inference
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)

        elif cfg.RCNN.ENABLED or cfg.IOUN.ENABLED:
            output = {}
            # rpn inference
            rcnn_output = self.rcnn_net(input_data)
            output.update(rcnn_output)
        else:
            raise NotImplementedError

        return output


    def rpn_forward(self, input_data):
        if cfg.RPN.ENABLED:
            output = {}
            # rpn inference
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)
        return output

    def rcnn_forward(self, rcnn_input_info):
        output = {}
        rcnn_output = self.rcnn_net(rcnn_input_info)
        output.update(rcnn_output)
        return output