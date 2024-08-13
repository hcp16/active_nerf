import json
import os

import cv2
import imageio
import numpy as np
import torch
import pickle
from PIL import Image
import PIL

def translate_by_t_along_z(t):
    tform = np.eye(4).astype(np.float32)
    tform[2][3] = t
    return tform


def rotate_by_phi_along_x(phi):
    tform = np.eye(4).astype(np.float32)
    tform[1, 1] = tform[2, 2] = np.cos(phi)
    tform[1, 2] = -np.sin(phi)
    tform[2, 1] = -tform[1, 2]
    return tform


def rotate_by_theta_along_y(theta):
    tform = np.eye(4).astype(np.float32)
    tform[0, 0] = tform[2, 2] = np.cos(theta)
    tform[0, 2] = -np.sin(theta)
    tform[2, 0] = -tform[0, 2]
    return tform


def pose_spherical(theta, phi, radius):
    c2w = translate_by_t_along_z(radius)
    c2w = rotate_by_phi_along_x(phi / 180.0 * np.pi) @ c2w
    c2w = rotate_by_theta_along_y(theta / 180 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_blender_data(basedir, half_res=False, sceneid="hotdog"):
    basedir = os.path.join(basedir, sceneid)

    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, s, f"transforms.json"), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_imgs_off = []
    all_poses = []
    all_intrinsics = []
    all_ir_poses = []
    all_normals = []
    all_depths = []
    all_labels = []
    all_names = []
    counts = [0]
    for s in splits:
        
        meta = metas[s]
        imgs = []
        imgs_off = []
        poses = []
        intrinsics = []
        ir_poses = []
        normals = []
        depths = []
        labels = []

        print(s, " ", len(meta["frames"]), " pictures")

        for frame in meta["frames"]:
            fname = os.path.join(basedir, s, frame["file_path"], "ir_on.png")
            fname_off = os.path.join(basedir, s, frame["file_path"], "ir_off.png")
            cimg = imageio.imread(fname)
            cimg_off = imageio.imread(fname_off)
            imgs.append(cimg)
            imgs_off.append(cimg_off)
            cpose = np.array(frame["transform_matrix"])
            poses.append(cpose)
            ir_poses.append(np.array(frame["transform_ir"]))

            fov = meta['camera_angle_x']
            width = cimg.shape[1]
            focal = .5 * width / np.tan(.5 * fov)
            cur_intrinsic = np.eye(3)
            cur_intrinsic[:2,2] = width/2
            cur_intrinsic[0,0] = focal
            cur_intrinsic[1,1] = focal

            if half_res:
                cur_intrinsic[:2,:] = cur_intrinsic[:2,:]/2
                intrinsics.append(cur_intrinsic)
            else:
                intrinsics.append(cur_intrinsic)

            normal_img = cv2.imread(os.path.join(basedir, s, frame["file_path"], "normal.png"), cv2.IMREAD_UNCHANGED)
            normal_img = (normal_img.astype(float)) / 1000. - 1

            norm = np.linalg.norm(normal_img, axis=-1)
            norm_mask = norm == 0
            dummy_normal = np.zeros([np.sum(norm_mask),3])
            dummy_normal[:,-1] = 1.
            normal_img[norm_mask] = dummy_normal
            normals.append(normal_img)

            cdepth = np.array(Image.open(os.path.join(basedir, s, frame["file_path"], "depth.png")))/1000.
            depths.append(cdepth)

            labelc = np.ones_like(cdepth)*18
            mask = np.logical_and(cdepth < 5.5,  cdepth > 0)
            labelc[mask] = 1
            labelc = labelc.astype(np.uint8)
            labels.append(labelc)

            all_names.append(fname)

        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        imgs_off = (np.array(imgs_off) / 255.0).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        ir_poses = np.array(ir_poses).astype(np.float32)
        intrinsics = np.array(intrinsics).astype(np.float32)
        normals = np.array(normals).astype(np.float32)
        depths = np.array(depths).astype(np.float32)
        labels = np.array(labels).astype(np.float32)

        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_imgs_off.append(imgs_off)
        all_poses.append(poses)
        all_intrinsics.append(intrinsics)
        all_ir_poses.append(ir_poses)
        all_normals.append(normals)
        all_depths.append(depths)
        all_labels.append(labels)


    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    imgs_off = np.concatenate(all_imgs_off, 0)
    poses = np.concatenate(all_poses, 0)
    ir_poses = np.concatenate(all_ir_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    normals = np.concatenate(all_normals, 0)
    depths = np.concatenate(all_depths, 0)
    labels = np.concatenate(all_labels, 0)

    H, W = imgs[0].shape[:2]

    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )

    if half_res:
        # TODO: resize images using INTER_AREA (cv2)
        H = H // 2
        W = W // 2
    imgs = [
        torch.from_numpy(
            cv2.resize(imgs[i], dsize=(W, H), interpolation=cv2.INTER_AREA)
        )
        for i in range(imgs.shape[0])
    ]
    imgs = torch.stack(imgs, 0)

    imgs_off = [
        torch.from_numpy(
            cv2.resize(imgs_off[i], dsize=(W, H), interpolation=cv2.INTER_AREA)
        )
        for i in range(imgs_off.shape[0])
    ]
    imgs_off = torch.stack(imgs_off, 0)

    normals = [
        torch.from_numpy(
            cv2.resize(normals[i], dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        )
        for i in range(normals.shape[0])
    ]
    normals = torch.stack(normals, 0)

    depths = [
        torch.from_numpy(
            cv2.resize(depths[i], dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        )
        for i in range(depths.shape[0])
    ]
    depths = torch.stack(depths, 0)

    labels = [
        torch.from_numpy(
            cv2.resize(labels[i], dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        )
        for i in range(labels.shape[0])
    ]
    labels = torch.stack(labels, 0)
    poses = torch.from_numpy(poses)
    ir_poses = torch.from_numpy(ir_poses)
    intrinsics = torch.from_numpy(intrinsics)

    return imgs, poses, ir_poses, render_poses, [H, W], i_split, intrinsics, depths, labels, imgs_off, normals, all_names


def load_d435_real(basedir, half_res=False, cfg=None, sceneid = "0"):
    lightdir = os.path.join(basedir, 'light_point.npy')
    intrdir = os.path.join(basedir, 'cam_intr.npy')
    basedir = os.path.join(basedir, sceneid)
    imgname = cfg.dataset.imgname
    imgname_off = cfg.dataset.imgname_off
    splits = ["train", "val", "test"]

    all_imgs = []
    all_poses = []
    all_ir_poses = []
    all_intrinsics = []
    all_imgs_off = []
    counts = [0]

    for s in splits:
        path = os.path.join(basedir, s)
        imgs = []
        poses = []
        intrinsics = []
        imgs_off = []
        ir_poses = []
        idx = 0
        prefix_list = os.listdir(path)
        prefix_list.sort(key=lambda a : int(a))
        for prefix in prefix_list:
            fname = os.path.join(path, prefix, imgname)
            fname_off = os.path.join(path, prefix, imgname_off)

            cur_img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            cur_img_off = cv2.imread(fname_off, cv2.IMREAD_UNCHANGED)
            imgs.append(cur_img)
            cur_pose = np.load(os.path.join(path, prefix, 'trans.npy'))
            cur_pose[:3,3] = cur_pose[:3,3]/1000.
            poses.append(cur_pose)

            cur_rel_ir_pose = np.load(lightdir)
            cur_rel_ir_pose = np.array([
                [1.,0.,0.,cur_rel_ir_pose[0]/1000.],
                [0.,1.,0.,cur_rel_ir_pose[1]/1000.],
                [0.,0.,1.,cur_rel_ir_pose[2]/1000.],
                [0.,0.,0.,1.]
            ])
            cur_ir_pose = cur_pose @ cur_rel_ir_pose

            ir_poses.append(cur_ir_pose)
            imgs_off.append(cur_img_off)
            ori_intrinsics = np.load(intrdir)
            if half_res:
                intrinsics_c = np.array(ori_intrinsics)
                intrinsics_c[:2, :] = intrinsics_c[:2, :] / 2
                intrinsics.append(intrinsics_c)
            else:
                intrinsics.append(np.array(ori_intrinsics))
            idx += 1

        poses = np.array(poses).astype(np.float32)
        ir_poses = np.array(ir_poses).astype(np.float32)
        intrinsics = np.array(intrinsics).astype(np.float32)
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        imgs_off = (np.array(imgs_off) / 255.0).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_ir_poses.append(ir_poses)
        all_intrinsics.append(intrinsics)
        all_imgs_off.append(imgs_off)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    ir_poses = np.concatenate(all_ir_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    imgs_off = np.concatenate(all_imgs_off,0)
    H, W = imgs[0].shape[:2]

    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )

    if half_res:
        H = H // 2
        W = W // 2

    imgs = [
        torch.from_numpy(
            cv2.resize(imgs[i], dsize=(W, H), interpolation=cv2.INTER_AREA)
        )
        for i in range(imgs.shape[0])
    ]
    imgs = torch.stack(imgs, 0)

    imgs_off = [
        torch.from_numpy(
            cv2.resize(imgs_off[i], dsize=(W, H), interpolation=cv2.INTER_AREA)
        )
        for i in range(imgs_off.shape[0])
    ]
    imgs_off = torch.stack(imgs_off, 0)

    poses = torch.from_numpy(poses)
    ir_poses = torch.from_numpy(ir_poses)
    intrinsics = torch.from_numpy(intrinsics)
    return imgs, poses, ir_poses, render_poses, [H, W], i_split, intrinsics, imgs_off

