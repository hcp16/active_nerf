import torch
import torch.nn.functional as F


class FlexibleNeRFModel(torch.nn.Module):
    '''
    vanilla NeRF MLP model
    input: position, (direction)
    output: rgb, sigma
    '''
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        color_channel=3
    ):
        super(FlexibleNeRFModel, self).__init__()

        self.num_layers = num_layers

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        #print(self.dim_xyz, self.dim_dir)
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        #torch.nn.init.zeros_(self.layer1.weight)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                #print(self.dim_xyz + hidden_size)
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, color_channel)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, color_channel+1)

        self.relu = torch.nn.functional.relu

    def forward(self, xin):
        if self.use_viewdirs:
            xyz, view = xin[..., : self.dim_xyz], xin[..., self.dim_xyz :]
        else:
            xyz = xin[..., : self.dim_xyz]
        
        x = self.layer1(xyz)
        
        for i in range(len(self.layers_xyz)):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != self.num_layers - 1
            ):
                #print("enter")
                x = torch.cat((x, xyz), dim=-1)
            #print(x.shape, len(self.layers_xyz), self.layers_xyz[i], i, self.num_layers - 1)
            x = self.relu(self.layers_xyz[i](x))
        
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)

            return torch.cat((rgb, alpha), dim=-1)
            
        else:
            return self.fc_out(x)


class FlexibleIRReflectanceModel(torch.nn.Module):
    '''
    MLP model for ir pattern
    explicit predict ir pattern
    '''
    def __init__(
        self,
        color_channel=3,
        H=1080,
        W=1920,
        ir_intrinsic=None,
        ir_extrinsic=None,
        ir_gt=None,
        #reflectence
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True
    ):
        super(FlexibleIRReflectanceModel, self).__init__()
        self.static_ir_pat = False

        self.H = 2*H
        self.W = 2*W
        self.ir_pattern = torch.nn.parameter.Parameter(torch.zeros([2*W,2*H]), requires_grad=True)
        if ir_gt is not None:
            self.ir_pattern = torch.load(ir_gt)
            self.ir_pattern.requires_grad = False
        self.ir_intrinsic = ir_intrinsic

        self.relu = torch.nn.functional.relu
        self.act_brdf = torch.nn.Sigmoid()

        self.num_layers = num_layers

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        #print(self.dim_xyz, self.dim_dir)
        self.skip_connect_every = skip_connect_every

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        #torch.nn.init.zeros_(self.layer1.weight)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                #print(self.dim_xyz + hidden_size)
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.layers_dir = torch.nn.ModuleList()
        # This deviates from the original paper, and follows the code release instead.
        self.layers_dir.append(
            torch.nn.Linear(2*self.dim_dir + hidden_size, hidden_size // 2)
        )

        self.fc_brdf = torch.nn.Linear(hidden_size // 2, 1)
        self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)


    def forward(self, xin):
        '''
        input: position, direction to camera, direction to light source
        output: direct light
        '''
        xyz, view, light = xin[..., : self.dim_xyz], xin[..., self.dim_xyz : self.dim_xyz + self.dim_dir],  xin[..., self.dim_xyz + self.dim_dir:]

        x = self.layer1(xyz)
        
        for i in range(len(self.layers_xyz)):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != self.num_layers - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))

        feat = self.relu(self.fc_feat(x))
        x = torch.cat((feat, view, light), dim=-1)
        for l in self.layers_dir:
            x = self.relu(l(x))
        brdf = self.fc_brdf(x)
        brdf = self.act_brdf(brdf)

        return brdf



    def get_light(self, surface_xyz, light_extrinsic, z):
        '''
        get ir pattern for a point
        Args:
            surface_xyz: point position under world coordinate
            light_extrinsic: ir light source under world coordinate
            z: surface point corresponding to input point

        Returns:
            light_out: ir pattern for each point
            surf2l: direction from light source to sampled points
        '''
        device=surface_xyz.device
        # world to ir light source
        w2ir = torch.linalg.inv(light_extrinsic)
        surface_xyz = surface_xyz.transpose(1,2) # n x 3 x s
        ir_xyz = torch.matmul(w2ir[:3,:3], surface_xyz) + w2ir[:3,3][...,None] # n x 3 x s

        # correspond to ir pixel
        irl_pix_homo = torch.matmul(self.ir_intrinsic, ir_xyz) # n x 3 x s
        irl_pix = (irl_pix_homo / irl_pix_homo[:,-1,:][:,None,:])[:,:2,:] # n x 2 x s
        irl_pix = irl_pix.transpose(1,2)[None,...]
        irl_pix[:,:,:,0] = ((irl_pix[:,:,:,0]/(self.W-1)) - 0.5) * 2.
        irl_pix[:,:,:,1] = ((irl_pix[:,:,:,1]/(self.H-1)) - 0.5) * 2.
        grid = torch.zeros(irl_pix.shape).to(device)
        grid[:,:,:,0] = irl_pix[:,:,:,1]
        grid[:,:,:,1] = irl_pix[:,:,:,0]

        # get ir value
        ir_pattern = torch.nn.functional.softplus(self.ir_pattern, beta=5)
        ir_pattern = ir_pattern[None,None,...]
        if self.static_ir_pat:
            max_ir = torch.max(ir_pattern)
            min_ir = torch.min(ir_pattern)
            mid_ir = 0.5*(max_ir + min_ir)
            light_out = F.grid_sample(ir_pattern, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
            light_out[light_out >= mid_ir] = max_ir
            light_out[light_out < mid_ir] = min_ir
            light_out = light_out[0,0,0,:]
        else:
            light_out = F.grid_sample(ir_pattern, grid, mode="bilinear", padding_mode="reflection", align_corners=True) # 1 x 1 x b x s
            light_out = light_out[0,0,:,:] # b x s

        # get the direction from light source to sampled points
        surf2l_ray = (light_extrinsic[:3,3][...,None] - z.transpose(0,1)).transpose(0,1)
        surf2l = F.normalize(surf2l_ray,p=2.0,dim=1)

        return light_out, surf2l

