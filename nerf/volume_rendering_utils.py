import torch
import torch.nn.functional as F

from .nerf_helpers import cumprod_exclusive, get_minibatches

import os


def run_network_ir_env(network_fn, pts, surf2c, surf2l, chunksize, embed_fn, embeddirs_fn):
    '''
    run network for model w/ active light
    Args:
        network_fn: model
        pts: points coordinate
        surf2c: ray origin under world coordinate
        surf2l: direction from light source to surface point corresponding to input point
        chunksize: mini-batch size
        embed_fn: encoding function for position
        embeddirs_fn: encoding function for direction

    Returns:

    '''
    # embedding
    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    if embeddirs_fn is not None:
        viewdirs = surf2c[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        camdirs = surf2l[..., None, -3:]
        output_dirs = camdirs.expand(pts.shape)
        output_dirs_flat = output_dirs.reshape((-1, output_dirs.shape[-1]))
        embedded_indirs = embeddirs_fn(input_dirs_flat)
        embedded_outdirs = embeddirs_fn(output_dirs_flat)
        embedded = torch.cat((embedded, embedded_indirs, embedded_outdirs), dim=-1)

    # split batches
    batches = get_minibatches(embedded, chunksize=chunksize)
    # forward the model for each mini-batch and concat the results
    preds = [network_fn(batch) for batch in batches]
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )
    return radiance_field


def volume_render_radiance_field_ir_env(
    radiance_field,
    depth_values,
    ray_origins,
    ray_directions,
    c_ray_directions,
    model_env=None,
    pts=None,
    radiance_field_noise_std=0.0,
    white_background=False,
    m_thres_cand=None,
    color_channel=3,
    idx=None,
    joint=False,
    is_env=False,
    mode="train",
    logdir=None,
    light_extrinsic=None,
    radiance_backup=None,
    encode_position_fn=None,
    encode_direction_fn=None,
    device=None,
    chunksize=131072
):
    '''
    volume render
    Args:
        radiance_field: output of MLP model, color (1 or 3 channel) and sigma
        depth_values: z value of sampled points
        ray_origins: ray origin under world coordinate
        ray_directions: ray direction under world coordinate
        c_ray_directions: ray direction under camera coordinate
        model_env: model for active light
        pts: ray direction under camera coordinate
        radiance_field_noise_std: add noise to sigma
        white_background: bool, True for white background
        m_thres_cand:
        color_channel: number of color channels
        idx: ray pixel index (uv)
        joint: bool, if True, image w/ active light will be also used to optimize model_fine
        is_env: bool, if True, loss of image w/ active light will be added
        mode: train or validation
        logdir: path of log
        light_extrinsic: light position under world coordinate
        radiance_backup: output of model_backup, the model of the previous step
        encode_position_fn: encoding function for position
        encode_direction_fn: encoding function for direction
        device: device of models

    Returns:
        tuple: rgb_map, env_rgb_map, surf_brdf, disp_map, acc_map, weights, depth_map, depth_map_max, depth_map_backup, sigma_a, depth_map_dex
    '''
    # calculate distance between sampled points
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = radiance_field[..., :color_channel]
    occupancy = radiance_field[..., color_channel]
    if not torch.all(~torch.isnan(rgb)):
        print("nan rgb!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    elif not torch.all(~torch.isnan(occupancy)):
        print("nan occupancy!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # get weight
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., color_channel].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
        noise = noise.to(radiance_field)
    sigma_a = torch.nn.functional.relu(radiance_field[..., color_channel] + noise)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)# bs x p
    acc_map = weights.sum(dim=-1)

    # rgb w/o active light
    rgb_map = None
    env_rgb = torch.sigmoid(rgb)
    surf_brdf = None
    env_rgb_map = weights[..., None] * env_rgb
    env_rgb_map = env_rgb_map.sum(dim=-2)
    if white_background:
        env_rgb_map = env_rgb_map + (1.0 - acc_map[..., None])

    # weighted depth
    depth_map = weights * depth_values
    depth_map = depth_map.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    # get depth of model_backup
    depth_map_backup = None
    if radiance_backup is not None:
        with torch.no_grad():
            sigma_a_b = torch.nn.functional.relu(radiance_backup[..., color_channel]) 
            alpha_b = 1.0 - torch.exp(-sigma_a_b * dists)
            weights_b = alpha_b * cumprod_exclusive(1.0 - alpha_b + 1e-10)# bs x p
            depth_map_backup = weights_b * depth_values
            depth_map_backup = depth_map_backup.detach()
            depth_map_backup = depth_map_backup.sum(dim=-1)

    # depth of Dex-NeRF
    depth_map_dex = []
    for m_thres in m_thres_cand:
        thres_out = (sigma_a > m_thres).type(torch.int)
        depth_ind = torch.argmax(thres_out, dim=-1)
        n_ind = torch.arange(depth_ind.shape[0])
        depth_map_dex.append(depth_values[n_ind, depth_ind])

    # depth of largest weight
    max_idx = torch.max(weights,dim=-1).indices # bs x 1
    depth_map_max = depth_values[list(range(depth_values.shape[0])),max_idx]

    if model_env is not None:  # predict ir
        # save results
        if mode == "test":
            if (os.path.exists(os.path.join(logdir, "weights.pt"))):
                weight_save = torch.load(os.path.join(logdir, "weights.pt"))
                depth_values_save = torch.load(os.path.join(logdir, "depth_values.pt"))
                radiance_field_save = torch.load(os.path.join(logdir, "occu.pt"))
                dists_save = torch.load(os.path.join(logdir, "dists.pt"))
                depth_map_save = torch.load(os.path.join(logdir, "depth_map.pt"))
                depth_map_max_save = torch.load(os.path.join(logdir, "depth_map_max.pt"))

                weight_save = torch.cat((weight_save, weights.cpu()), 0)
                depth_values_save = torch.cat((depth_values_save, depth_values.cpu()), 0)
                radiance_field_save = torch.cat((radiance_field_save, radiance_field.cpu()), 0)
                dists_save = torch.cat((dists_save, dists.cpu()), 0)
                depth_map_save = torch.cat((depth_map_save, depth_map.cpu()), 0)
                depth_map_max_save = torch.cat((depth_map_max_save, depth_map_max.cpu()), 0)

                torch.save(weight_save, os.path.join(logdir, "weights.pt"))
                torch.save(depth_values_save, os.path.join(logdir, "depth_values.pt"))
                torch.save(radiance_field_save, os.path.join(logdir, "occu.pt"))
                torch.save(dists_save, os.path.join(logdir, "dists.pt"))
                torch.save(depth_map_save, os.path.join(logdir, "depth_map.pt"))
                torch.save(depth_map_max_save, os.path.join(logdir, "depth_map_max.pt"))
            else:
                torch.save(weights.cpu(), os.path.join(logdir, "weights.pt"))
                torch.save(depth_values.cpu(), os.path.join(logdir, "depth_values.pt"))
                torch.save(radiance_field.cpu(), os.path.join(logdir, "occu.pt"))
                torch.save(dists.cpu(), os.path.join(logdir, "dists.pt"))
                torch.save(depth_map.cpu(), os.path.join(logdir, "depth_map.pt"))
                torch.save(depth_map_max.cpu(), os.path.join(logdir, "depth_map_max.pt"))

        # get ir pattern for each point
        rays_o = ray_origins
        rays_d = F.normalize(ray_directions,p=2.0,dim=1)
        surface_z = depth_map
        surface_xyz = rays_o + (surface_z).unsqueeze(-1) * rays_d  # [bs, 3]
        if joint == True:
            direct_light, surf2l = model_env.get_light(pts.detach(), light_extrinsic, surface_xyz.detach())  # ir pattern, bs x 3
            direct_light = torch.sum(direct_light*weights, dim=-1)[...,None]
        else:
            direct_light, surf2l = model_env.get_light(pts.detach(), light_extrinsic, surface_xyz.detach()) # bs x 3
            direct_light = torch.sum(direct_light*(weights.detach()), dim=-1)[...,None]
        surf2c = -rays_d
        
        # get ir reflection for each point
        radiance_field_env = torch.ones(radiance_field.shape[:2]).unsqueeze(-1).to(device)
        if is_env:
            radiance_field_env = run_network_ir_env(
                model_env,
                pts,  # bs x s x 3
                surf2c, # bs x 3
                surf2l, # bs x 3
                chunksize,
                encode_position_fn,
                encode_direction_fn,
            )

        # predict captured ir
        if joint == True:
            surf_brdf = weights[...,None] * radiance_field_env
        else:
            surf_brdf = weights[...,None].detach() * radiance_field_env
        surf_brdf = surf_brdf.sum(dim=-2)
        rgb_ir = direct_light * surf_brdf  # [bs, 1]

        # add ir and environment rgb as the prediction of image w/ active light
        if joint == True:
            rgb_map = env_rgb_map + rgb_ir
        else:
            rgb_map = env_rgb_map.detach() + rgb_ir
        rgb_map = torch.clip(rgb_map,0.,1.)

    out = [rgb_map, env_rgb_map, surf_brdf, disp_map, acc_map, weights, depth_map, depth_map_max, depth_map_backup, sigma_a] + depth_map_dex
    return tuple(out)

