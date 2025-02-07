import torch
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.body_model.lbs import lbs

from util.utils_rot import rvecs_2_rot_mat

class SMPLXModel(BodyModel):
    def __init__(self, device=None, **kwargs):
        super().__init__(**kwargs)
        self.device = device if device is not None else torch.device('cuda')
        for name in ['init_pose_hand', 'init_pose_jaw','init_pose_eye', 'init_v_template', 'init_expression', 
                    'shapedirs', 'exprdirs', 'posedirs', 'J_regressor', 'kintree_table', 'weights', ]:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(device))
        
    def forward(self, pose_body, betas, trans=None, use_rodrigues=False):
        batch_size = pose_body.shape[0]
        
        pose_hand = self.init_pose_hand.expand(batch_size, -1)
        pose_jaw = self.init_pose_jaw.expand(batch_size, -1)
        pose_eye = self.init_pose_eye.expand(batch_size, -1)
        v_template = self.init_v_template.expand(batch_size, -1, -1)
        expression = self.init_expression.expand(batch_size, -1)

        init_pose = torch.cat([pose_jaw, pose_eye, pose_hand], dim=-1)
        if not use_rodrigues:
            init_pose = rvecs_2_rot_mat(init_pose)
        #print(pose_body.shape)
        #print(init_pose.shape)
        #print(v_template.shape)
        #print(expression.shape)
        #print(betas.shape)
        #exit()
        full_pose = torch.cat([pose_body, init_pose], dim=-1)
        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.exprdirs], dim=-1)

        verts, joints = lbs(betas=shape_components, pose=full_pose, v_template=v_template,
                        shapedirs=shapedirs, posedirs=self.posedirs, J_regressor=self.J_regressor,
                        parents=self.kintree_table[0].long(), lbs_weights=self.weights, pose2rot=use_rodrigues)
        if trans is not None:
            joints = joints + trans.unsqueeze(dim=1)
            verts = verts + trans.unsqueeze(dim=1)
        return dict(verts=verts, joints=joints, faces=self.f)