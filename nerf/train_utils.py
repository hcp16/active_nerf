import torch
import torch.nn.functional as F
import numpy as np
import copy

from .nerf_helpers import get_minibatches, ndc_rays
from .nerf_helpers import sample_pdf_2 as sample_pdf
from .volume_rendering_utils import volume_render_radiance_field_ir_env

def gen_error_colormap_img():
    cols = np.array(
        [[0, 0.00001, 0, 0, 0],
         [0.00001, 4./(2**10) , 49, 54, 149],
         [4./(2**10) , 4./(2**9) , 69, 117, 180],
         [4./(2**9) , 4./(2**8) , 116, 173, 209],
         [4./(2**8), 4./(2**7), 171, 217, 233],
         [4./(2**7), 4./(2**6), 224, 243, 248],
         [4./(2**6), 4./(2**5), 254, 224, 144],
         [4./(2**5), 4./(2**4), 253, 174, 97],
         [4./(2**4), 4./(2**3), 244, 109, 67],
         [4./(2**3), 4./(2**2), 215, 48, 39],
         [4./(2**2), np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols


def render_error_img(R_est_tensor, R_gt_tensor, mask, abs_thres=1.):
    R_gt_np = R_gt_tensor.detach().cpu().numpy()
    R_est_np = R_est_tensor.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    H, W = R_gt_np.shape
    # valid mask
    # mask = (D_gt_np > 0) & (D_gt_np < 1250)
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(R_gt_np - R_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = error[mask] / abs_thres
    # get colormap
    cols = gen_error_colormap_img()
    # create error image
    error_image = np.zeros([H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
    # TODO: imdilate
    # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));
    error_image[np.logical_not(mask)] = 0.
    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:10, i * distance:(i + 1) * distance, :] = cols[i, 2:]
    return error_image # [H, W, 3]


def run_network_ir(network_fn, pts, ray_batch, chunksize, embed_fn, embeddirs_fn):
    '''
    run model
    Args:
        network_fn: model
        pts: points B*M*3
        ray_batch: rays B*3
        chunksize: mini-batch size
        embed_fn: encoding function for position
        embeddirs_fn: encoding function for direction
    Returns:
        radiance_field: output of the model
    '''
    # embedding
    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)
    # split batches
    batches = get_minibatches(embedded, chunksize=chunksize)

    # forward the model for each mini-batch and concat the results
    preds = [network_fn(batch) for batch in batches]
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )
    return radiance_field



def predict_and_render_radiance_ir(
    ray_batch,
    model_coarse,
    model_fine,
    model_env_fine,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    m_thres_cand=None,
    joint=False,
    is_env=False,
    logdir=None,
    light_extrinsic=None,
    is_rgb=False,
    model_backup=None,
    device=None
):
    '''
    Args:
        ray_batch: a batch of all the inputs N*16. 0~2: ray_origins; 3~5: ray_directions; 6~8: cam_origins; 9~11: cam_directions; 12: near; 13: far; 14~15: idx
        model_coarse: coarse-resolution model w/o active light
        model_fine: fine-resolution model w/o active light
        model_env_fine: model for active light
        options: config
        mode: train or validation
        encode_position_fn: encoding function for position
        encode_direction_fn: encoding function for direction
        m_thres_cand:
        joint: if True, image w/ active light will be also used to optimize model_fine
        is_env: if True, loss of image w/ active light will be added
        logdir: path of log
        light_extrinsic: light position under world coordinate
        is_rgb: if False, use one-channel image
        model_backup: model of the previous step used to calculate loss to constraint output change, set None to ignore the loss
        device: device of models

    Returns:
        tuple: rgb_coarse, rgb_off_coarse, disp_coarse, acc_coarse, rgb_fine, rgb_off_fine, disp_fine, acc_fine, brdf_fine, depth_fine_nerf, depth_fine_nerf_max, depth_fine_nerf_backup, weights_fine
    '''
    num_rays = ray_batch.shape[0]
    ro, rd, c_ro, c_rd = ray_batch[..., :3], ray_batch[..., 3:6], ray_batch[..., 6:9], ray_batch[..., 9:12]
    bounds = ray_batch[..., 12:14].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]
    idx = ray_batch[...,14:16].type(torch.long)

    # sample points on each ray
    t_vals = torch.linspace(
        0.0,
        1.0,
        getattr(options.nerf, mode).num_coarse,
        dtype=ro.dtype,
        device=ro.device,
    )
    if not getattr(options.nerf, mode).lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])
    if getattr(options.nerf, mode).perturb:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
    c_pts = c_ro[..., None, :] + c_rd[..., None, :] * z_vals[..., :, None]

    # run model_coarse
    radiance_field = run_network_ir(
        model_coarse,
        pts,
        ray_batch[..., -6:-3],
        getattr(options.nerf, mode).chunksize,
        encode_position_fn,
        encode_direction_fn
    )

    # render model_coarse
    coarse_out = volume_render_radiance_field_ir_env(
        radiance_field,
        z_vals,
        ro,
        rd,
        c_rd,
        None,
        radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
        white_background=getattr(options.nerf, mode).white_background,
        m_thres_cand=m_thres_cand,
        color_channel=3 if is_rgb else 1,
        idx=idx,
        joint=joint,
        light_extrinsic=light_extrinsic,
        device=device,
        chunksize=getattr(options.nerf, mode).chunksize
    )
    rgb_coarse, rgb_off_coarse, disp_coarse, acc_coarse, weights, depth_coarse = coarse_out[0], coarse_out[1], coarse_out[3], coarse_out[4], coarse_out[5], coarse_out[6]

    rgb_fine, disp_fine, acc_fine = None, None, None
    if getattr(options.nerf, mode).num_fine > 0:  # model_fine
        # importance sampling
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(options.nerf, mode).num_fine,
            det=(getattr(options.nerf, mode).perturb == 0.0),
        )
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)

        pts_fine = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
        c_pts_fine = c_ro[..., None, :] + c_rd[..., None, :] * z_vals[..., :, None]

        # run model_fine
        radiance_field = run_network_ir(
            model_fine,
            pts_fine,
            ray_batch[..., -6:-3],
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
        )

        # run model_backup
        radiance_field_backup = None
        if model_backup is not None:
            radiance_field_backup = run_network_ir(
                model_backup,
                pts_fine,
                ray_batch[..., -6:-3],
                getattr(options.nerf, mode).chunksize,
                encode_position_fn,
                encode_direction_fn,
            )

        # run and render model_env_fine
        fine_out = volume_render_radiance_field_ir_env(
            radiance_field,
            z_vals,
            ro,
            rd,
            c_rd,
            model_env_fine,
            pts=pts_fine,
            radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
            m_thres_cand=m_thres_cand,
            color_channel=3 if is_rgb else 1,
            idx=idx,
            joint=joint,
            is_env=is_env,
            mode=mode,
            logdir=logdir,
            light_extrinsic=light_extrinsic,
            radiance_backup=radiance_field_backup,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            device=device,
            chunksize=getattr(options.nerf, mode).chunksize
        )
        rgb_fine, rgb_off_fine, brdf_fine, disp_fine, acc_fine = fine_out[0], fine_out[1], fine_out[2], fine_out[3], fine_out[4]
        weights_fine = fine_out[5]
        depth_fine_nerf = fine_out[6]
        depth_fine_nerf_max = fine_out[7]
        depth_fine_nerf_backup = fine_out[8]
        depth_fine_dex = list(fine_out[10:])

    out = [rgb_coarse, rgb_off_coarse, disp_coarse, acc_coarse, \
        rgb_fine, rgb_off_fine, disp_fine, acc_fine, \
        brdf_fine, depth_fine_nerf, depth_fine_nerf_max, \
        depth_fine_nerf_backup, weights_fine\
        ] + depth_fine_dex
    return tuple(out)



def run_one_iter_of_nerf_ir(
    height,
    width,
    focal_length,
    model_coarse,
    model_fine,
    model_env_fine,
    ray_origins,
    ray_directions,
    cam_origins,
    cam_directions,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    m_thres_cand=None,
    idx=None,
    joint=False,
    is_env=False,
    logdir=None,
    light_extrinsic=None,
    is_rgb=False,
    model_backup=None,
    device=None
):
    '''
    run NeRF
    Args:
        height: float, image height
        width: float, image width
        focal_length: float, camera focal length
        model_coarse: coarse-resolution model w/o active light
        model_fine: fine-resolution model w/o active light
        model_env_fine: model for active light
        ray_origins: ray origin under world coordinate
        ray_directions: ray direction under world coordinate
        cam_origins: ray origin under camera coordinate
        cam_directions: ray direction under camera coordinate
        options: config
        mode: train or validation
        encode_position_fn: encoding function for position
        encode_direction_fn: encoding function for direction
        m_thres_cand:
        idx: ray pixel index (uv)
        joint: if True, image w/ active light will be also used to optimize model_fine
        is_env: if True, loss of image w/ active light will be added
        logdir: path of log
        light_extrinsic: light position under world coordinate
        is_rgb: if False, use one-channel image
        model_backup: model of the previous step used to calculate loss to constraint output change, set None to ignore the loss
        device: device of models

    Returns:
        tuple: rgb_coarse, rgb_off_coarse, disp_coarse, acc_coarse, rgb_fine, rgb_off_fine, disp_fine, acc_fine, brdf_fine, depth_fine_nerf, depth_fine_nerf_max, depth_fine_nerf_backup, weights_fine
    '''
    viewdirs = None
    
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))

        cam_viewdirs = cam_directions
        cam_viewdirs = cam_viewdirs / cam_viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        cam_viewdirs = cam_viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    out_shape = ray_directions[...,0].unsqueeze(-1).shape
    if is_rgb:
        out_shape = ray_directions.shape

    restore_shapes = [
        out_shape,
        out_shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
    ]
    if model_fine:
        restore_shapes += restore_shapes
        restore_shapes += [ray_directions.shape[:-1]] # brdf_fine
        restore_shapes += [ray_directions.shape[:-1]] # depth_fine
        restore_shapes += [ray_directions.shape[:-1]] # depth_fine_max
        restore_shapes += [ray_directions.shape[:-1]] # depth_fine_backup
        H,W = ray_directions.shape[0],ray_directions.shape[1]
        restore_shapes += [torch.Size([H,W,128])] # weights
        for _ in m_thres_cand:
            restore_shapes += [ray_directions.shape[:-1]]
    if options.dataset.no_ndc is False:
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
        c_ro, c_rd = ndc_rays(height, width, focal_length, 1.0, cam_origins, cam_directions)
        c_ro = c_ro.view((-1, 3))
        c_rd = c_rd.view((-1, 3))
    else:
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.view((-1, 3))
        c_ro = cam_origins.view((-1, 3))
        c_rd = cam_directions.view((-1, 3))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, c_ro, c_rd, near, far, idx), dim=-1)
    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs, cam_viewdirs), dim=-1)

    # split input into mini-batches to avoid memory explosion
    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    pred = [
        # render one mini-batch
        predict_and_render_radiance_ir(
            batch,
            model_coarse,
            model_fine,
            model_env_fine,
            options,
            mode=mode,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            m_thres_cand=m_thres_cand,
            joint=joint,
            is_env=is_env,
            logdir=logdir,
            light_extrinsic=light_extrinsic,
            is_rgb=is_rgb,
            model_backup=model_backup,
            device=device
        )
        for batch in batches
    ]

    # concat all the mini-batches
    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
    if mode == "validation" or mode == "test":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are set to None.
            return tuple(synthesized_images + [None, None, None])

    return tuple(synthesized_images)


