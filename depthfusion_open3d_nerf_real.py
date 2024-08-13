import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import pickle
from PIL import Image
import argparse
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_path", type=str, default="./logs/elephant")
    parser.add_argument("--dilate", type=int, default=0)
    parser.add_argument("--mask_path", type=str, default="./demos/elephant/train")
    parser.add_argument("--mask_name", type=str, default="sam.png")
    parser.add_argument("--meta_prefix", type=str, default="validation_meta_")
    parser.add_argument("--depth_prefix", type=str, default="validation_pred_depth_step_250000_")
    parser.add_argument("--depth_file_path", type=str, default="pred_depth_nerf_max")
    parser.add_argument("--meta_file_path", type=str, default="meta")
    parser.add_argument("--output_path", type=str, default="./logs/elephant/pcd_max.ply")
    parser.add_argument('--use_gpu', action='store_false')
    opt = parser.parse_args()
    return opt


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def compute_pcd(depth_in, intrinsics, extrinsics, device):
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight'), attr_dtypes=(o3d.core.float32, o3d.core.float32), attr_channels=((1), (1)), voxel_size=1.0 / 512, block_resolution=16, block_count=50000, device=device)

    depth_scale = 1000.0
    depth_max = 1.5

    for depth, extrinsic, intrinsic in tqdm(zip(depth_in, extrinsics, intrinsics)):
        depth = depth.to(device)
        frustum_block_coords = vbg.compute_unique_block_coordinates(depth, intrinsic, extrinsic, depth_scale, depth_max)
        vbg.integrate(frustum_block_coords, depth, intrinsic, extrinsic, depth_scale, depth_max)
    pcd = vbg.extract_point_cloud()
    pcd = pcd.to_legacy()
    return pcd


if __name__ == '__main__':
    opt = parse_args()
    logs_path = opt.logs_path
    mask_path = opt.mask_path
    mask_name = opt.mask_name
    meta_prefix = opt.meta_prefix
    depth_prefix = opt.depth_prefix
    output_path = opt.output_path
    depth_file_path = os.path.join(logs_path, opt.depth_file_path)
    meta_file_path = os.path.join(logs_path, opt.meta_file_path)
    dilate = opt.dilate

    if opt.use_gpu:
        device = o3d.core.Device('cuda:0')
    else:
        device = o3d.core.Device('cpu:0')

    meta_path = os.path.join(logs_path, 'split.pkl')
    meta = load_pickle(meta_path)
    i_train = meta['i_train']

    intrinsics = []
    extrinsics = []
    depth_pred = []
    depth_nerf_pred = []
    depth_dex = []
    depth_gt = []
    for i in i_train:
        meta_path = os.path.join(meta_file_path, meta_prefix + str(i) + '.pkl')
        if not os.path.exists(meta_path):
            print('missing meta', i, meta_path)
            continue
        meta = load_pickle(meta_path)
        
        pred_depth_fname = os.path.join(depth_file_path, depth_prefix + str(i) + '.png')
        if not os.path.exists(pred_depth_fname):
            print('missing depth', i, pred_depth_fname)
            continue
        depth = np.array(Image.open(pred_depth_fname))
        H, W = depth.shape

        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, meta['intrinsic'])
        intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix, dtype=o3d.core.Dtype.Float64)
        intrinsics.append(intrinsic)

        extrinsic = np.array(meta['extrinsic'])
        extrinsic = np.linalg.inv(extrinsic)
        extrinsic = o3d.core.Tensor(extrinsic, dtype=o3d.core.Dtype.Float64)
        extrinsics.append(extrinsic)

        mask_fname = os.path.join(mask_path, str(i), mask_name)
        if not os.path.exists(mask_fname):
            print('missing mask', i, mask_fname)
            continue
        mask = np.array(Image.open(mask_fname).resize([W, H]))
        if dilate > 0:
            mask = (mask > 0).astype(np.uint8)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate)))
        mask = (mask == 0)

        depth[mask] = 4000
        depth = depth.astype(np.uint16)
        depth = o3d.t.geometry.Image(depth)
        depth_pred.append(depth)

    pcd = compute_pcd(depth_pred, intrinsics, extrinsics, device)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(output_path, pcd)

