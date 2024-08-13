# ActiveNeRF: Learning Accurate 3D Geometry by Active Pattern Projection
### [[Paper]]()
This repository contains the official implementation (in Pytorch) for "ActiveNeRF: Learning Accurate 3D Geometry by Active Pattern Projection" . The part of nerf is based on https://github.com/krrish94/nerf-pytorch. 

<!-- ### Citation -->


## Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data](#data)
4. [Training](#training)
5. [Depth Fusion](#depth-fusion)

## Introduction
NeRFs have achieved incredible success in novel view synthesis. However, the accuracy of the implicit geometry is unsatisfactory because the passive static environmental illumination has low spatial frequency and cannot provide enough information for accurate geometry reconstruction. In this work, we propose ActiveNeRF, a 3D geometry reconstruction framework, which improves the geometry quality of NeRF by actively projecting patterns of high spatial frequency onto the scene using a projector which has a constant relative pose to the camera. We design a learnable active pattern rendering pipeline which jointly learns the scene geometry and the active pattern. We find that, by adding the active pattern and imposing its consistency across different views, our proposed method outperforms state of the art geometry reconstruction methods qualitatively and quantitatively in both simulation and real experiments. 

## Installation
### Dependencies
- imageio
- numpy
- opencv-python-headless
- pyyaml
- tensorboard
- tqdm
- open3d
- git+https://github.com/aliutkus/torchsearchsorted


## Data
```
<data folder name>
│  cam_intr.npy     intrinsic parameters of the camera
│  light_point.npy  light point location under the camera coordinate system
│
└─<scene id>
        ├─test
        │  ├─<test view 0>
        │  │      img_ir_off.png    image with ir off
        │  │      img_ir_on.png     image with ir on
        │  │      sam.png           segmentation
        │  │      trans.npy         transformation matrix from world to the view
        │  │
        │  └─<test view 1>
        │          ...
        │
        ├─train
        │  ├─<train view 0>
        │  │      img_ir_off.png
        │  │      img_ir_on.png
        │  │      sam.png
        │  │      trans.npy
        │  │
        │  └─<train view 1>
        │         ...
        │
        └─val
            └─<validation view 0>
                    ...
```

## Training
We provide an example of training script in `configs/brdf-real_elephant.yml`

To train with one GPU, run following command in the terminal:
```
python train_nerf_ir2_real.py --config config/brdf-real_elephant.yml --sceneid elephant
```

## Depth Fusion
To get the point cloud of the object, we first render the depth map under each training view and fuse them with VoxelBlockGrid of Open3D. 

To fuse the depth maps in the training output, run following command in the terminal:

```
python depthfusion_open3d_nerf_real.py --logs_path ./logs/elephant/2024_08_13__15_09_41 --output_path ./logs/elephant/2024_08_13__15_09_41/pcd_max.ply
```
