import numpy as np
import torch
from util.smpl_model import SMPLXModel
from util.utils_rot import rvecs_2_rot_mat, rotation6d_2_rot_mat, rvecs_2_rotation6d, rotation6d_2_rvecs
from util.visual import EvaluateStreamPlot
import os
from body_model.body_model import BodyModel
from .viz.utils import viz_smpl_seq
from scipy.spatial.transform import Rotation

def mesh_generator(pred_verts, label_verts, faces):
    result = dict(
        pred_smpl = dict(
            mesh = [pred_verts, faces],
            color = np.asarray([255, 160, 160]) / 255  #light red
        ),
        label_smpl = dict(
            mesh = [label_verts, faces],
            color = np.asarray([160, 210, 255]) / 255  #light blue
        )
    )
    yield result

def Item(value):
    return value.detach().cpu()

def visualize_motion_in_smpl_2Models(joint_pre, joint_gt, shapeInfo):
    save_path = "./save_smpl_video/"
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    body_model = SMPLXModel(bm_fname='./smpl/models_lockedhead/smplx/SMPLX_NEUTRAL.npz', num_betas=16, num_expressions=0, device=device)
    vis = EvaluateStreamPlot(save_path = save_path)
    #vis.init_show()
    
    yPred = rotation6d_2_rot_mat(torch.tensor(joint_pre).to(device))
    yGT = rotation6d_2_rot_mat(torch.tensor(joint_gt).to(device))
    yPred = yPred.unsqueeze(0)
    yGT = yGT.unsqueeze(0)

    img_array = []
    for i in range(100):
        beta = torch.tensor(shapeInfo).to(device).unsqueeze(0).float()
        pred_mesh = body_model(yPred[0][i:i+1], beta)
        label_mesh = body_model(yGT[0][i:i+1], beta)
        gen = mesh_generator(Item(pred_mesh['verts'][0]), Item(label_mesh['verts'][0]), Item(pred_mesh['faces']))
        vis.show(gen, fps=300)
        print(i)

NUM_BETAS = 16
def get_body_model_sequence(smplh_path, gender, num_frames,
                  pose_body, pose_hand, betas, root_orient, trans):
    gender = str(gender)
    bm_path = os.path.join(smplh_path, gender + '/model.npz')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bm = BodyModel(bm_path=bm_path, num_betas=NUM_BETAS, batch_size=num_frames).to(device)

    #pose_body = torch.Tensor(pose_body).to(device)
    #pose_hand = torch.Tensor(pose_hand).to(device)
    betas = torch.Tensor(np.repeat(betas[:NUM_BETAS][np.newaxis], num_frames, axis=0)).to(device)
    #root_orient = torch.Tensor(root_orient).to(device)
    #trans = torch.Tensor(trans).to(device)
    #trans = trans + torch.Tensor(np.array([0., .15, .0])).to(device)
    body = bm(pose_body=pose_body, pose_hand=pose_hand, betas=betas, root_orient=root_orient, trans=trans)
    return body

def visualize_motion_in_smpl_global(joint_pre, shapeInfo, world2aligned_rot = None, trans = None, joint_traj = None, body = None, waypoints = None, use_KIT_Skeleton = None):
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    print(joint_pre.shape)
    if joint_pre.shape[-1] == 3:
        yPred = torch.tensor(joint_pre).to(device).to(torch.float32)
    else:
        yPred = rotation6d_2_rvecs(torch.tensor(joint_pre).reshape((-1, 6)).to(device))
    yPred = yPred.reshape((-1, 22, 3))
    
    num_frames = yPred.shape[0]
    if world2aligned_rot is not None:
        root_orient_mats = Rotation.from_rotvec(yPred[:, 0, :].cpu()).as_matrix()
        world2aligned_rot = world2aligned_rot[0:1, :, :]
        world2aligned_rot = np.repeat(world2aligned_rot, root_orient_mats.shape[0], axis = 0)
        
        root_orient_mats = np.einsum("ijk, ikl->ijl", world2aligned_rot, root_orient_mats)
        root_orient = Rotation.from_matrix(root_orient_mats).as_rotvec()
        yPred[:, 0, :] = torch.Tensor(root_orient).to(device)
        trans = np.einsum("ijk, ik->ij", world2aligned_rot, trans)
    pose_body = yPred[:, 1:, :].reshape((-1, 63))
    root_orient = yPred[:, 0, :]
    
    
    trans = torch.Tensor(trans).to(device) #+ torch.Tensor(np.array([.0, .0, 1.0])).to(device)  #add extra height to trans
    '''
        print(pose_body.shape)    # (frames, 63)
        print(pose_hand.shape)    # (frames, 90)
        print(root_orient.shape)  # (frames, 3)
        print(trans.shape)        # (frames, 3)
        print(shapeInfo.shape)    # (16,)
    '''
    pose_hand = torch.zeros((num_frames, 90)).to(device)
    shapeInfo = np.zeros((16,))
    #print(joint_traj[:, 3, :])
    if body is None:
        body = get_body_model_sequence("./body_models/smplh", "male", num_frames,
                                pose_body, pose_hand, shapeInfo, root_orient, trans) 
    if joint_traj is not None:
        if use_KIT_Skeleton is not None:
            viz_smpl_seq(body, imw=1080, imh=1080, fps=10, contacts=None, skel_connections = use_KIT_Skeleton,
                    render_body=False, render_joints=True, render_skeleton=True, render_ground=True,
                    joints_seq=joint_traj, joint_rad=0.0225, points_seq = waypoints)
        else:
            viz_smpl_seq(body, imw=1080, imh=1080, fps=10, contacts=None,
                    render_body=False, render_joints=True, render_skeleton=True, render_ground=True,
                    joints_seq=joint_traj, joint_rad=0.0225, points_seq = waypoints)
    else:
        viz_smpl_seq(body, imw=1080, imh=1080, fps=10, contacts=None,
                render_body=True, render_joints=False, render_skeleton=False, render_ground=True,
                joints_seq=None, joint_rad=0.0225)