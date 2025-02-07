from models.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
import torch
from visualize.simplify_loc2rot import joints2smpl

class npy2obj:
    def __init__(self, npy_path, sample_idx, rep_idx, device=0, cuda=True):
        self.npy_path = npy_path
        self.motions = np.load(self.npy_path, allow_pickle=True)
        self.motions['motion'] = self.motions['motion'].numpy()
        self.start_root_pos = self.motions['motion'][0, 0, :, :]
        
        
        self.my_motion = torch.tensor(self.motions['motion'][..., :]) # DO NOT comment this line. Change Time only
        #self.motions['motion'] =self.motions['motion'][..., :20] #For debug only
        
        if self.npy_path.endswith('.npz'):
            self.motions = self.motions['arr_0']
        #self.motions = self.motions[None][0]
        self.rot2xyz = Rotation2xyz(device='cpu')
        self.faces = self.rot2xyz.smpl_model.faces
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.opt_cache = {}
        self.sample_idx = sample_idx
        self.total_num_samples = self.motions['num_samples']
        self.rep_idx = rep_idx
        self.absl_idx = self.rep_idx*self.total_num_samples + self.sample_idx
        self.num_frames = self.motions['motion'][self.absl_idx].shape[-1]
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)
        
        #print(self.motions['num_samples'])
        #print(self.motions['lengths'])
        #print(self.motions['motion'].shape)
        #exit()

        if self.nfeats == 3:
            print(f'Running SMPLify For sample [{sample_idx}], repetition [{rep_idx}], it may take a few minutes.')
            motion_tensor, opt_dict = self.j2s.joint2smpl(self.motions['motion'][self.absl_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
            self.motions['motion'] = motion_tensor.cpu().numpy()
        elif self.nfeats == 6:
            self.motions['motion'] = self.motions['motion'][[self.absl_idx]]
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.real_num_frames = self.motions['lengths'][self.absl_idx]
        #self.real_num_frames = 20

        self.vertices = self.rot2xyz(torch.tensor(self.motions['motion']), mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='vertices',
                                     # jointstype='smpl',  # for joint locations
                                     vertstrans=True)
        
        
        temp_joint = self.rot2xyz(torch.tensor(self.motions['motion']), mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='smpl',
                                     # jointstype='smpl',  # for joint locations
                                     vertstrans=True)
        self.displacement = self.my_motion[0,0,:,:] - temp_joint[0,0,:,:] # B, 24, 3 , T
        self.vertices   += self.displacement
        self.joints   = temp_joint+ self.displacement
        
        self.to_be_saved_SMPL_joint = temp_joint + self.displacement
        
        # print(self.to_be_saved_SMPL_joint[0].permute(2,0,1)[:,:22,:].shape)
        # input(222)
        #self.root_loc = self.motions['motion'][:, -1, :3, :].reshape(1, 1, 3, -1)
        
        #self.root_loc = self.start_root_pos.reshape(1, 1, 3, -1)
        
        # self.vertices += self.root_loc
        
        

    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i),
                       faces=self.faces)

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path
    
    def save_skel(self, save_path, frame_i):
        
        parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        with open(save_path, 'w') as obj_file:
            my_joint = self.joints[0].permute(2,0,1).cpu().numpy()
            for i in range(len(my_joint[frame_i])):
                x, y, z = my_joint[frame_i][i]
                obj_file.write(f"v {x} {y} {z}\n")
            for i, i_parent in enumerate(parents):
                if i_parent == -1:
                    continue
                obj_file.write(f"l {i+1} {i_parent+1}\n")

    def save_motion(self, save_path, frame_i):
        parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        with open(save_path, 'w') as obj_file:
            my_motion = torch.tensor(self.my_motion)[0].permute(2,0,1).cpu().numpy()
            for i in range(len(my_motion[frame_i])):
                x, y, z = my_motion[frame_i][i]
                obj_file.write(f"v {x} {y} {z}\n")
            for i, i_parent in enumerate(parents):
                if i_parent == -1:
                    continue
                obj_file.write(f"l {i+1} {i_parent+1}\n")
    
    def save_npy(self, save_path):
        
        data_dict = {
            'motion': self.motions['motion'][0, :, :, :self.real_num_frames],
            'thetas': self.motions['motion'][0, :-1, :, :self.real_num_frames],
            'root_translation': self.motions['motion'][0, -1, :3, :self.real_num_frames],
            'faces': self.faces,
            'vertices': self.vertices[0, :, :, :self.real_num_frames],
            'text': None,
            'length': self.real_num_frames,
            'control_traj': self.to_be_saved_SMPL_joint[0].permute(2,0,1)[:,:22,:]
        }
        np.save(save_path, data_dict)
