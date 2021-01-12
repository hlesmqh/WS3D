# WS3D

## Weakly Supervised 3D object detection from Lidar Point Cloud

This is the official repo of 'Weakly Supervised 3D object detection from Lidar Point Cloud' (ECCV2020).<br/><br/>
**Author**: Qinghao Meng, [Wenguan Wang](https://sites.google.com/view/wenguanwang), Tianfei Zhou, Jianbing Shen, Luc Van Gool, and Dengxin Dai

![intro](https://github.com/hlesmqh/WS3D/blob/master/intro.png)

## Introduction：
This work proposes a weakly supervised approach for 3D object detection, only requiring a small set of weakly annotated scenes, associated with a few precisely labeled object instances. This is achieved by a two-stage architecture design. Using only 500 weakly annotated scenes and 534 precisely labeled vehicle instances, our method achieves 85−95% the performance of current top-leading, fully supervised detectors (which require 3, 712 exhaustively and precisely annotated scenes with 15, 654 instances) on KITTI 3D object detection [leaderboard](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). More importantly, our trained model can be applied as a 3D object annotator, generating annotations which can be used to train 3D object detectors with over 94% of their original performance (under manually labeled data). Above designs make our approach highly practical and introduce new opportunities for learning 3D object detection with reduced annotation burden.

For more details of WS3D, please refer to [our paper](https://arxiv.org/abs/2007.11901v1) or [project page](#).
The implementation is based on the preexisting open source codebase [PointRCNN](https://github.com/sshaoshuai/PointRCNN).

### ToDo list
- [x] Installation
- [x] Dataset preparation
- [x] BEV annotator instruction
- [x] BEV center-click annotation
- [x] Stage-1 Training
- [x] Partly labeled objects list
- [x] Stage-2 data preparation
- [x] Stage-2 Training
- [x] 3D Annotation tool instruction
- [x] Pretrained model 

## Installation:

### Requirements:
All the codes are tested in the following environment:

```Linux (tested on Ubuntu 16.04)```, 
```Python 3.6```
```PyTorch 1.1.0```

### Install WS3D

a. Clone the PointRCNN repository.
```shell
git clone --recursive https://github.com/hlesmqh/WS3D.git
```

b. Install the dependent python libraries.
```shell
pip install -r requirement.txt 
```

c. Build and install the `pointnet2_lib`, `iou3d`, `roipool3d` libraries by executing the following command:
```shell
sh build_and_install.sh
```

## Dataset Preparation
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

## BEV Annotator Instruction

Our BEV center click annotator is placed in `/Pointcloud_Annotation/`. For running annotator, you should run:
```shell
python ./Pointcloud_Annotation/annotation.py 
```
Be aware that you need to have Qt interface accessible on your machine.

## BEV center-click annotation
Our BEV click annotation can be get from [here](https://drive.google.com/file/d/1A-HWh6HFq9JFp4IJghWfbOs80a8ergPt/view?usp=sharing) or [BaiduDisk](https://github.com/hlesmqh/WS3D/blob/master/baidudownloadlink.txt).


## Stage-1 Training
```shell
python ./tools/train_rpn.py --noise_kind='label_noise' --weakly_num=500
```
- The `noise_kind` is the directory of BEV center-click annotation file, it is saved as KITTI official Label format, but only (x,z) information available.
- The `weakly_num` is the number of click annotated scenes, in our implementation, we choose the first 500 non-empty scenes in KITTI training split, which is already officially random shuffled.
- The other training parameter can be found in file `tools/cfgs/weaklyRPN` and in args of `/tools/train_rpn.py`.
- Our BEV annotator and BEV center-click annotation will available soon, but you can also set `noise_kind='lable_2'` for using accurate (x,z) information from KITTI original label.

## Stage-2 Data Perparation
Please select your trained stage-1 model and generate your stage-2 training set following below guidence:
Change  ```/tools/generate_box_dataset.py```
```shell
ckpt_file = '/path/to/your/ckpt.pth'
save_dir =  '/path/to/save/this/small/trainingset/'
```
The program will generates a file saving proposals according to the result of your stage-1 model and saves them with nearby groundtruth boxes.

## Partly labeled objects list
This list is gained by randomly select groundtruth boxes which have at least one proposal nearby. For convenience, we write a script which help you select training instances from stage-2 training set. Our best model's list can be get by generating stage-2 training set by our pretrained [stage-1](https://drive.google.com/file/d/1RgsANIPsnYh1rctTifSzGSdG9v6_sS5F/view?usp=sharing) model. When you trains your stage-2 model, the program will selects them from your saved set. It will not be changed for next training. 

## Stage-2 Training
You need to change the training set path in ```self.boxes_dir = os.path.join(self.imageset_dir, 'boxes_410fl030500_Car')``` and then run:
```shell
python ./tools/train_cascade1.py --weakly_num=500
```

## Pretrained Model
You could download the pretrained model(Car) of WS3D from here [Stage-1](https://drive.google.com/file/d/1RgsANIPsnYh1rctTifSzGSdG9v6_sS5F/view?usp=sharing) and [Stage-2](https://drive.google.com/file/d/1V_pu3xDRwybNho6L7f0dxY3Qdvyz0pAG/view?usp=sharing), which is trained on the *train* split (3712 samples) and evaluated on the *val* split (3769 samples) and *test* split (7518 samples). The performance on validation set is as follows:
```
Car AP@0.70, 0.70, 0.70:
bbox AP:90.38, 89.15, 88.59
bev  AP:88.95, 85.83, 85.03
3d   AP:85.04, 75.94, 74.38
aos  AP:90.25, 88.78, 88.11
```

## 3D Annotation tool instruction

run ```python ./Pointcloud_Annotation/annotation```, and you can see the interface below.

![anno](https://github.com/hlesmqh/WS3D/blob/master/annotation.png)

Please first click the object on above camera view image. The program will select the nearest point projected on this view to you mouse and show a zoom in BEV map left below. If you aren't satisified with this region, you can click the camera view again for a better BEV region.
Then, you can click the BEV center of object on this zoom in BEV map, your click location will be saved in desired file which is setted at ```f = open('label_w/label.txt', 'a+')```. After labeling all objects you can click on global bev map right below for opening next scene. Please notice that the program will automatic start from you last labeled image.

Citation:
---------------

Please consider citing this paper if it helps your research:

    @inproceedings{meng2020ws3d,
        title={Weakly Supervised 3D Object Detection from Lidar Point Cloud},
        author={Meng, Qinghao and Wang, Wenguan and Zhou, Tianfei and Shen, Jianbing and Van Gool, Luc and Dai, Dengxin},
        booktitle={ECCV},
        year={2020}
    }
    

