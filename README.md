WS3D
==================================================
Weakly Supervised 3D object detection from Lidar Point Cloud
-------------------------------------------------------------
This is official version of 'Weakly Supervised 3D object detection from Lidar Point Cloud'(ECCV2020).<br/><br/>
**Author**: Qinghao Meng, [Wenguan Wang](https://sites.google.com/view/wenguanwang), Tianfei Zhou, Jianbing Shen, Luc Van Gool, and Dengxin Dai

![](https://github.com/hlesmqh/WS3D/blob/master/intro.png?raw=true)

### Introduction：
This work proposes a weakly supervised approach for 3D object detection, only requiring a small set of weakly annotated scenes, associated with a few precisely labeled object instances. This is achieved by a two-stage architecture design. Using only 500 weakly annotated scenes and 534 precisely labeled vehicle instances, our method achieves 85−95% the performance of current top-leading, fully supervised detectors (which require 3, 712 exhaustively and precisely annotated scenes with 15, 654 instances). More importantly, our trained model can be applied as a 3D object annotator, generating annotations which can be used to train 3D object detectors with over 94% of their original performance (under manually labeled data). Above designs make our approach highly practical and introduce new opportunities for learning 3D object detection with reduced annotation burden.

==========================================================
### Installation:
#### Requirements:
All the codes are tested in the following environment:

Linux (tested on Ubuntu 16.04)
Python 3.6+
PyTorch 1.0

==========================================================
### Citation:
Please cite these papers in your publications if it helps your research:
@inproceedings{meng2020ws3d,
  title={Weakly Supervised 3D Object Detection from Lidar Point Cloud},
  author={Meng, Qinghao and Wang, Wenguan and Zhou, Tianfei and Shen, Jianbing, Van Gool, Luc and Dai, Dengxin},
  booktitle={ECCV},
  year={2020}
}


**The code will be released soon.**
