import argparse
import os
import time
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import torchvision.utils as vutils
from PIL import Image
import open3d as o3d
import pickle
import datetime

from nerf import render_error_img

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse, load_blender_data, meshgrid_xy, models, mse2psnr, run_one_iter_of_nerf_ir, load_d435_real)

debug_output = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    parser.add_argument(
        "--sceneid",
        type=str,
        default="elephant",
        help="The scene id that need to train",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="d435_real",
        help="The datatype",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="The gpu ID",
    )
    opt = parser.parse_args()
    return opt

def main():
    configargs = parse_args()
    # Read config file.
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # Load dataset
    if configargs.data == 'blender':
        images, poses, ir_poses, render_poses, hw, i_split, intrinsics, depths, labels, imgs_off, normals, name = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            sceneid = configargs.sceneid
        )
        inverse_yz = True
    elif configargs.data == 'd435_real':
        images, poses, ir_poses, render_poses, hw, i_split, intrinsics, imgs_off = load_d435_real(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            cfg=cfg,
            sceneid = configargs.sceneid
        )
        inverse_yz = False
    else:
        raise ValueError('configargs.data')
    
    color_ch = 3 if cfg.dataset.is_rgb else 1
    i_train, i_val, i_test = i_split
    H, W = hw
    H, W = int(H), int(W)
    if cfg.nerf.train.white_background:
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{configargs.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    # position encoder for position
    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )
    # position encoder for direction (in and out)
    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize MLP models
    # Initialize a coarse-resolution model w/o active light
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_layers=cfg.models.coarse.num_layers,
        hidden_size=cfg.models.coarse.hidden_size,
        skip_connect_every=cfg.models.coarse.skip_connect_every,
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
        color_channel=color_ch
    )
    model_coarse.to(device)

    # Initialize a fine-resolution model w/o active light.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_layers=cfg.models.fine.num_layers,
            hidden_size=cfg.models.fine.hidden_size,
            skip_connect_every=cfg.models.fine.skip_connect_every,
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
            color_channel=color_ch
        )
        model_fine.to(device)

    # Initialize a model for active light.
    model_env_fine = None
    if not cfg.dataset.is_rgb:
        ir_intrinsic = intrinsics[0,:,:].to(device)
        ir_intrinsic[:2,2] = ir_intrinsic[:2,2] * 2.
        ir_intrinsic[:2,:2] = ir_intrinsic[:2,:2] * 2.
        ir_extrinsic = ir_poses[0,:,:].to(device)

        model_env_fine = getattr(models, cfg.models.env.type)(
            color_channel=1,
            H = H,
            W = W,
            ir_intrinsic=ir_intrinsic,
            ir_extrinsic=ir_extrinsic,
            num_layers=cfg.models.env.num_layers,
            hidden_size=cfg.models.env.hidden_size,
            skip_connect_every=cfg.models.env.skip_connect_every,
            num_encoding_fn_xyz=cfg.models.env.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.env.num_encoding_fn_dir,
            include_input_xyz=cfg.models.env.include_input_xyz,
            include_input_dir=cfg.models.env.include_input_dir,
        )
        model_env_fine.to(device)

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    trainable_parameters += list(model_fine.parameters())
    if not cfg.dataset.is_rgb:
        trainable_parameters += list(model_env_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )

    # Setup logging.
    now = datetime.datetime.now()
    logdir = os.path.join(cfg.experiment.logdir, str(configargs.sceneid), now.strftime('%Y_%m_%d__%H_%M_%S'))
    os.makedirs(logdir, exist_ok=True)
    m_thres_max = cfg.nerf.validation.m_thres
    m_thres_cand = np.arange(5,m_thres_max+5,1)
    os.makedirs(os.path.join(logdir,"pred_depth_dex"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_depth_err_dex"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_depth_nerf"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_depth_nerf_max"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_depth_gt"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_depth_err_nerf"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_depth_err_nerf_max"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_nerf"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_nerf_gt"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_depth_pcd_nerf"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_depth_pcd_nerf_gt"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_depth_pcd_nerf_max"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"meta"), exist_ok=True)

    # Write out config parameters.
    writer = SummaryWriter(logdir)
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)
    split_info = {
        'i_train': i_train,
        'i_val': i_val,
        'i_test': i_test,
    }
    with open(os.path.join(logdir, "split.pkl"), "wb") as f:
        pickle.dump(split_info, f)

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_iter = 0

    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        if not cfg.dataset.is_rgb:
            model_env_fine.load_state_dict(checkpoint["model_env_fine_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]

    # initialize state of parameters
    is_joint = False
    is_env = False
    for param in model_coarse.parameters():
        param.requires_grad = True
    for param in model_fine.parameters():
        param.requires_grad = True
    for param in model_env_fine.parameters():
        param.requires_grad = True
    model_backup = None

    # training
    for i in trange(start_iter, cfg.experiment.train_iters):
        # When is_joint == True, the loss of image w/ active light will be used to optimize model_env_fine and model_fine. When is_joint == False, the loss of image w/ active light will only be used to optimize model_env_fine.
        # When is_joint == True, the loss of active image will be added.
        if i == cfg.experiment.jointtrain_start:
            is_joint = True
        if i == cfg.experiment.finetune_start:            
            is_env = True

        # load data to GPU
        img_idx = np.random.choice(i_train)
        img_off_target = imgs_off[img_idx].to(device)  # image w/o active light
        img_target = images[img_idx].to(device)  # image w/ active light
        pose_target = poses[img_idx, :, :].to(device)  # camera pose
        intrinsic_target = intrinsics[img_idx,:,:].to(device)  # camera intrinsic
        ir_extrinsic_target = ir_poses[img_idx,:,:].to(device)  # light source pose

        # get rays for each pixel
        ray_origins, ray_directions, cam_origins, cam_directions = get_ray_bundle(H, W, pose_target, intrinsic_target, inverse_yz)

        # get sample indices
        coords = torch.stack(
            meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
            dim=-1,
        )
        coords = coords.reshape((-1, 2))
        select_inds = np.random.choice(
            coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
        )
        select_inds = coords[select_inds]

        # sample rays and pixels
        ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
        ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
        cam_origins = cam_origins.to(device)
        cam_origins = cam_origins[select_inds[:, 0], select_inds[:, 1], :]
        cam_directions = cam_directions[select_inds[:, 0], select_inds[:, 1], :]
        target_s = img_target[select_inds[:, 0], select_inds[:, 1]] # [1080]
        target_s_off = img_off_target[select_inds[:, 0], select_inds[:, 1]]

        # run nerf
        nerf_out = run_one_iter_of_nerf_ir(
            H,
            W,
            intrinsic_target[0,0],
            model_coarse,
            model_fine,
            model_env_fine,
            ray_origins,
            ray_directions,
            cam_origins.to(device),
            cam_directions.to(device),
            cfg,
            mode="train",
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            m_thres_cand=m_thres_cand,
            idx=select_inds,
            joint=is_joint,
            is_env=is_env,
            logdir=logdir,
            light_extrinsic=ir_extrinsic_target,
            is_rgb=cfg.dataset.is_rgb,
            model_backup=model_backup,
            device=device
        )
        
        rgb_coarse, rgb_off_coarse, rgb_fine, rgb_off_fine = nerf_out[0], nerf_out[1], nerf_out[4], nerf_out[5]
        brdf_fine = nerf_out[8]
        depth_fine_nerf = nerf_out[9]
        depth_fine_nerf_backup = nerf_out[11]
        
        # loss of images w/o active light
        target_ray_values = target_s.unsqueeze(-1)
        target_ray_values_off = target_s_off.unsqueeze(-1)
        coarse_loss_off = torch.nn.functional.mse_loss(
            torch.squeeze(rgb_off_coarse), torch.squeeze(target_ray_values_off)
        )
        fine_loss_off = torch.nn.functional.mse_loss(
            torch.squeeze(rgb_off_fine), torch.squeeze(target_ray_values_off)
        )

        fine_loss = 0.
        if not cfg.dataset.is_rgb:
            # loss of images w/ active light
            fine_loss = torch.nn.functional.mse_loss(
                    rgb_fine, target_ray_values
            )

        if not torch.all(~torch.isnan(fine_loss)):
            print("nan fineloss")
            return

        # loss
        loss_off = coarse_loss_off + fine_loss_off
        loss_on = fine_loss
        loss = cfg.experiment.ir_on_rate * loss_on + \
            cfg.experiment.ir_off_rate * loss_off

        # calculate PSNR
        if cfg.dataset.is_rgb:
            psnr = mse2psnr(fine_loss_off.item())
        else:
            psnr = mse2psnr(fine_loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Learning rate updates
        num_decay_steps = cfg.scheduler.lr_decay * 1000
        lr_new = cfg.optimizer.lr * (
            cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

        # write log
        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write(
                "[TRAIN] Iter: "
                + str(i)
                + " Loss: "
                + str(loss.item())
                + " PSNR: "
                + str(psnr)
            )
        writer.add_scalar("train/loss", loss.item(), i)
        writer.add_scalar("train/coarse_loss_off", coarse_loss_off.item(), i)
        writer.add_scalar("train/fine_loss_off", fine_loss_off.item(), i)
        if not cfg.dataset.is_rgb:
            writer.add_scalar("train/fine_loss", fine_loss.item(), i)
            writer.add_scalar("train/psnr", psnr, i)

        if i == 0:
            continue
        # Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1
        ):
            tqdm.write("[VAL] =======> Iter: " + str(i))
            test_mode = "validation"
            model_coarse.eval()
            model_fine.eval()
            if not cfg.dataset.is_rgb:
                model_env_fine.eval()

            start = time.time()
            loss_list = []
            psnr_list = []
            for img_idx in tqdm(np.concatenate((i_train, i_val, i_test))):
                #tqdm.write(str(img_idx))
                with torch.no_grad():
                    img_target = images[img_idx].to(device)
                    pose_target = poses[img_idx, :, :].to(device)
                    ir_extrinsic_target = ir_poses[img_idx,:,:].to(device)
                    img_off_target = imgs_off[img_idx].to(device)
                    intrinsic_target = intrinsics[img_idx,:,:].to(device)
                    ray_origins, ray_directions, cam_origins, cam_directions = get_ray_bundle(
                        H, W, pose_target, intrinsic_target, inverse_yz
                    )
                    coords = torch.stack(
                        meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
                        dim=-1,
                    )
                    
                    coords = coords.permute(1,0,2)
                    coords = coords.reshape((-1, 2))

                    nerf_out = run_one_iter_of_nerf_ir(
                        H,
                        W,
                        intrinsic_target[0,0],
                        model_coarse,
                        model_fine,
                        model_env_fine,
                        ray_origins,
                        ray_directions,
                        cam_origins.to(device),
                        cam_directions.to(device),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                        m_thres_cand=m_thres_cand,
                        idx = coords,
                        joint=is_joint,
                        is_env=is_env,
                        logdir=logdir,
                        light_extrinsic=ir_extrinsic_target,
                        is_rgb=cfg.dataset.is_rgb,
                        device=device
                    )
                    rgb_coarse, rgb_coarse_off, rgb_fine, rgb_fine_off = nerf_out[0], nerf_out[1], nerf_out[4], nerf_out[5]
                    brdf_fine = nerf_out[8]
                    depth_fine_nerf = nerf_out[9]
                    depth_fine_nerf_max = nerf_out[10]
                    depth_fine_dex = list(nerf_out[13:])
                    target_ray_values = img_target.unsqueeze(-1)
                    target_ray_values_off = img_off_target.unsqueeze(-1)

                    # calculate fine loss w/o active light
                    if img_idx in i_test:
                        loss, fine_loss = 0.0, 0.0
                        if not cfg.dataset.is_rgb:
                            if rgb_fine is not None:
                                fine_loss = img2mse(rgb_fine, target_ray_values)
                                loss = fine_loss
                            else:
                                #loss = coarse_loss
                                loss = img2mse(rgb_coarse, target_ray_values)
                        else:
                            fine_loss = img2mse(torch.squeeze(rgb_fine_off), torch.squeeze(target_ray_values_off))
                            loss = fine_loss
                        psnr = mse2psnr(loss.item())
                        loss_list.append(loss.item())
                        psnr_list.append(psnr)

                    img_ground_mask = torch.ones(depth_fine_nerf.shape).type(torch.bool)

                    # save rgb w/o active light
                    rgb_fine_np = rgb_fine.cpu().numpy()[:,:,0]
                    img_target_np = img_target.cpu().numpy()
                    rgb_fine_np = (rgb_fine_np*255).astype(np.uint8)
                    img_target_np = (img_target_np*255).astype(np.uint8)
                    rgb_fine_np_img = Image.fromarray(rgb_fine_np, mode='L')
                    img_target_np_img = Image.fromarray(img_target_np, mode='L')
                    rgb_fine_np_img.save(os.path.join(logdir, "pred_nerf", test_mode + "_pred_nerf_step_" + str(i) + "_" + str(img_idx) + ".png"))
                    img_target_np_img.save(os.path.join(logdir, "pred_nerf_gt", test_mode + "_pred_nerf_gt_step_" + str(i) + "_" + str(img_idx) + ".png"))

                    # save depth
                    # weighted depth
                    pred_depth_nerf = depth_fine_nerf.detach().cpu()
                    pred_depth_nerf_max = depth_fine_nerf_max.detach().cpu()
                    pred_depth_nerf_np = pred_depth_nerf.numpy()
                    pred_depth_nerf_max_np = pred_depth_nerf_max.numpy()
                    pred_depth_nerf_np = pred_depth_nerf_np * 1000
                    pred_depth_nerf_np = (pred_depth_nerf_np).astype(np.uint32)
                    out_pred_depth_nerf = Image.fromarray(pred_depth_nerf_np, mode='I')
                    out_pred_depth_nerf.save(os.path.join(logdir, "pred_depth_nerf", test_mode + "_pred_depth_step_" + str(i) + "_" + str(img_idx) + ".png"))

                    # depth of max weight
                    pred_depth_nerf_max_np = pred_depth_nerf_max_np * 1000
                    pred_depth_nerf_max_np = (pred_depth_nerf_max_np).astype(np.uint32)
                    out_pred_depth_nerf_max = Image.fromarray(pred_depth_nerf_max_np, mode='I')
                    out_pred_depth_nerf_max.save(os.path.join(logdir, "pred_depth_nerf_max", test_mode + "_pred_depth_step_" + str(i) + "_" + str(img_idx) + ".png"))

                    # detailed output
                    if debug_output:
                        # depth to point cloud
                        depth_pts = depth2pts_np(pred_depth_nerf_np, intrinsic_target.cpu().numpy(), pose_target.cpu().numpy())
                        pts_o3d = o3d.utility.Vector3dVector(depth_pts)
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = pts_o3d
                        o3d.io.write_point_cloud(os.path.join(logdir,"pred_depth_pcd_nerf",test_mode+"_pred_depth_pcd_step_"+str(i)+ "_" + str(img_idx) + ".ply"), pcd)

                        depth_pts = depth2pts_np(pred_depth_nerf_max_np, intrinsic_target.cpu().numpy(), pose_target.cpu().numpy())
                        pts_o3d = o3d.utility.Vector3dVector(depth_pts)
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = pts_o3d
                        o3d.io.write_point_cloud(os.path.join(logdir,"pred_depth_pcd_nerf_max",test_mode+"_pred_depth_pcd_step_"+str(i)+ "_" + str(img_idx) + ".ply"), pcd)

                        # scene
                        scene_info = {
                            "intrinsic": intrinsic_target.cpu().numpy(),
                            "extrinsic": pose_target.cpu().numpy(),
                        }
                        meta_loc = os.path.join(logdir, "meta")
                        with open(os.path.join(meta_loc, test_mode+"_meta_" + str(img_idx) + ".pkl"), "wb") as f:
                            pickle.dump(scene_info, f)

            # save loss and psnr of test views to tensorboard
            loss_list = np.asarray(loss_list)
            psnr_list = np.asarray(psnr_list)
            writer.add_scalar(test_mode + "/loss", np.mean(loss_list), i)
            writer.add_scalar(test_mode + "/psnr", np.mean(psnr_list), i)

            # save images to tensorboard
            if rgb_fine_off is not None:
                if not cfg.dataset.is_rgb:
                    # ir pattern
                    ir_light = model_env_fine.ir_pattern.clone().detach()
                    ir_light_out = torch.nn.functional.softplus(ir_light, beta=5)
                    writer.add_image(
                        test_mode+"/ir_light", vutils.make_grid(ir_light_out, padding=0, nrow=1, normalize=True), i
                    )

                    # save predicted images w/ and w/o active light
                    writer.add_image(
                        test_mode+"/rgb_fine", vutils.make_grid(rgb_fine[...,0], padding=0, nrow=1), i
                    )
                    writer.add_image(
                        test_mode+"/rgb_fine_off", vutils.make_grid(rgb_fine_off[...,0], padding=0, nrow=1), i
                    )

                    writer.add_image(
                        test_mode+"/rgb_coarse_off", vutils.make_grid(rgb_coarse_off[...,0], padding=0, nrow=1), i
                    )
                    writer.add_image(
                        test_mode+"/rgb_brdf", vutils.make_grid(brdf_fine, padding=0, nrow=1), i
                    )

                else:
                    writer.add_image(
                        test_mode+"/rgb_fine_off", vutils.make_grid(rgb_fine_off[...,:].permute(2,0,1), padding=0, nrow=1), i
                    )
                    writer.add_image(
                        test_mode+"/rgb_coarse_off", vutils.make_grid(rgb_coarse_off[...,:].permute(2,0,1), padding=0, nrow=1), i
                    )
                writer.add_scalar(test_mode+"/fine_loss", fine_loss.item(), i)

            # save gt images
            writer.add_image(
                test_mode+"/img_target",
                vutils.make_grid(target_ray_values[...,0], padding=0, nrow=1),
                i,
            )
            if not cfg.dataset.is_rgb:
                writer.add_image(
                    test_mode+"/img_off_target",
                    vutils.make_grid(img_off_target, padding=0, nrow=1),
                    i,
                )
            else:
                writer.add_image(
                    test_mode+"/img_off_target",
                    vutils.make_grid(img_off_target.permute(2,0,1), padding=0, nrow=1),
                    i,
                )

            # save image error
            if not cfg.dataset.is_rgb:
                pred_rgb_fine_err_np = render_error_img(rgb_fine[...,0], target_ray_values[...,0], img_ground_mask)
                pred_rgb_fine_off_err_np = render_error_img(rgb_fine_off[...,0], img_off_target, img_ground_mask)


                writer.add_image(
                    test_mode+"/rgb_fine_err",
                    pred_rgb_fine_err_np.transpose((2,0,1)),
                    i,
                )
                writer.add_image(
                    test_mode+"/rgb_fine_off_err",
                    pred_rgb_fine_off_err_np.transpose((2,0,1)),
                    i,
                )

            writer.add_image(
                    test_mode+"/depth_pred_nerf",
                    vutils.make_grid(pred_depth_nerf, padding=0, nrow=1, normalize=True, scale_each=True),
                    i,
                )

            tqdm.write(
                "Validation loss: "
                + str(loss.item())
                + " Validation PSNR: "
                + str(psnr)
                + " Time: "
                + str(time.time() - start)
            )
            with open(os.path.join(logdir, test_mode+"_output_result.txt"), "a") as f:
                f.write("iter: "
                + str(i)
                + " Validation loss: "
                + str(loss.item())
                + " Validation PSNR: "
                + str(psnr)
                + " Time: "
                + str(time.time() - start)
                + "\n"
                )

        # save checkpoint
        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            if not cfg.dataset.is_rgb:
                checkpoint_dict = {
                    "iter": i,
                    "model_coarse_state_dict": model_coarse.state_dict(),
                    "model_fine_state_dict": model_fine.state_dict(),
                    "model_env_fine_state_dict": model_env_fine.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "psnr": psnr
                }
            else:
                checkpoint_dict = {
                    "iter": i,
                    "model_coarse_state_dict": model_coarse.state_dict(),
                    "model_fine_state_dict": model_fine.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "psnr": psnr
                }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")

    print("Done!")


def depth2pts_np(depth_map, cam_intrinsic, cam_extrinsic=np.eye(4)):
    feature_grid = get_pixel_grids_np(depth_map.shape[0], depth_map.shape[1])

    uv = np.matmul(np.linalg.inv(cam_intrinsic), feature_grid)
    cam_points = uv * np.reshape(depth_map, (1, -1))

    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3:4]
    R_inv = np.linalg.inv(R)

    world_points = np.matmul(R_inv, cam_points - t).transpose()
    return world_points


def get_pixel_grids_np(height, width):
    x_linspace = np.linspace(0.5, width - 0.5, width)
    y_linspace = np.linspace(0.5, height - 0.5, height)
    x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
    x_coordinates = np.reshape(x_coordinates, (1, -1))
    y_coordinates = np.reshape(y_coordinates, (1, -1))
    ones = np.ones_like(x_coordinates).astype(float)
    grid = np.concatenate([x_coordinates, y_coordinates, ones], axis=0)

    return grid


if __name__ == "__main__":
    main()
