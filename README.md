# WS3D

## Weakly Supervised 3D object detection from Lidar Point Cloud

This is the official repo of 'Weakly Supervised 3D object detection from Lidar Point Cloud' (ECCV2020).<br/><br/>
**Author**: Qinghao Meng, [Wenguan Wang](https://sites.google.com/view/wenguanwang), Tianfei Zhou, Jianbing Shen, Luc Van Gool, and Dengxin Dai

![teaser](https://github.com/hlesmqh/WS3D/blob/master/intro.png)

## Introduction：
This work proposes a weakly supervised approach for 3D object detection, only requiring a small set of weakly annotated scenes, associated with a few precisely labeled object instances. This is achieved by a two-stage architecture design. Using only 500 weakly annotated scenes and 534 precisely labeled vehicle instances, our method achieves 85−95% the performance of current top-leading, fully supervised detectors (which require 3, 712 exhaustively and precisely annotated scenes with 15, 654 instances) on KITTI 3D object detection [leaderboard](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). More importantly, our trained model can be applied as a 3D object annotator, generating annotations which can be used to train 3D object detectors with over 94% of their original performance (under manually labeled data). Above designs make our approach highly practical and introduce new opportunities for learning 3D object detection with reduced annotation burden.

For more details of WS3D, please refer to [our paper](https://arxiv.org/abs/2007.11901v1) or [project page](#).
The implementation is based on the preexisting open source codebase [PointRCNN](https://github.com/sshaoshuai/PointRCNN).

### ToDo list
- [x] Installation
- [x] Dataset preparation
- [ ] BEV annotator instruction
- [ ] BEV center-click annotation
- [x] Stage-1 Training
- [ ] Partly labeled objects list
- [ ] Stage-2 data preparation
- [ ] Stage-2 Training
- [ ] 3D Annotation tool instruction
- [ ] Pretrained model 

## Installation:

### Requirements:
All the codes are tested in the following environment:

```Linux (tested on Ubuntu 16.04)```, 
```Python 3.6```
```PyTorch 1.0```

### Install WS3D

a. Clone the PointRCNN repository.
```shell
git clone --recursive https://github.com/hlesmqh/WS3D.git
```

b. Install the dependent python libraries.
```shell
pip install requirements.txt -r
```

c. Build and install the `pointnet2_lib`, `iou3d`, `roipool3d` libraries by executing the following command:
```shell
sh build_and_install.sh
```

## Dataset preparation
Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows:
```
Kitti
├── ImageSets
├── object
│   ├──training
      ├──calib & velodyne & label_2 & image_2
│   ├──testing
│      ├──calib & velodyne & image_2
```

Change the the files ```/tools/train_*.py```  follow:
```shell
DATA_PATH = os.path.join('/your/path/Kitti/object')
```

## Stage-1 Training
```shell
python ./tools/train_rpn.py --noise_kind='label_noise' --weakly_num=500
```
- The `noise_kind` is the directory of BEV center-click annotation file, it is saved as KITTI official Label format, but only (x,z) information available.
- The `weakly_num` is the number of click annotated scenes, in our implementation, we choose the first 500 non-empty scenes in KITTI training split, which is already officially random shuffled.
- The other training parameter can be found in file `tools/cfgs/weaklyRPN` and in args of `/tools/train_rpn.py`.
- Our BEV annotator and BEV center-click annotation will available soon, but you can also set `noise_kind='lable_2'` for using accurate (x,z) information from KITTI original label.


## Pretrained model
You could download the pretrained model(Car) of WS3D from [here(release soon)](#), which is trained on the *sub-train* split (500 scenes in *train* split and 534 vehicles) and evaluated on the *val* split (3769 samples) and *test* split (7518 samples). 

Citation:
---------------

Please consider citing this paper if it helps your research:

    @inproceedings{meng2020ws3d,
        title={Weakly Supervised 3D Object Detection from Lidar Point Cloud},
        author={Meng, Qinghao and Wang, Wenguan and Zhou, Tianfei and Shen, Jianbing and Van Gool, Luc and Dai, Dengxin},
        booktitle={ECCV},
        year={2020}
    }
    
        
                      
**The code will be released soon.**


