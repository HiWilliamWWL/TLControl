import torch
from torch import nn
import torch.nn.functional as F

def rvecs_2_rot_mat(rvecs):
    batch_size = rvecs.shape[0]
    r_vecs = rvecs.reshape(-1, 3)
    total_size = r_vecs.shape[0]
    thetas = torch.norm(r_vecs, dim=1, keepdim=True)
    is_zero = torch.eq(torch.squeeze(thetas), torch.tensor(0.0))
    u = r_vecs / thetas

    # Each K is the cross product matrix of unit axis vectors
    # pyformat: disable
    zero = torch.autograd.Variable(torch.zeros([total_size], device=rvecs.device))  # for broadcasting
    Ks_1 = torch.stack([  zero   , -u[:, 2],  u[:, 1] ], axis=1)  # row 1
    Ks_2 = torch.stack([  u[:, 2],  zero   , -u[:, 0] ], axis=1)  # row 2
    Ks_3 = torch.stack([ -u[:, 1],  u[:, 0],  zero    ], axis=1)  # row 3
    # pyformat: enable
    Ks = torch.stack([Ks_1, Ks_2, Ks_3], axis=1)                  # stack rows

    identity_mat = torch.autograd.Variable(torch.eye(3, device=rvecs.device).repeat(total_size,1,1))
    Rs = identity_mat + torch.sin(thetas).unsqueeze(-1) * Ks + \
         (1 - torch.cos(thetas).unsqueeze(-1)) * torch.matmul(Ks, Ks)
    # Avoid returning NaNs where division by zero happened
    R = torch.where(is_zero[:,None,None], identity_mat, Rs)

    return R.reshape(batch_size, -1)


def rotation6d_2_rot_mat(rotation6d):
    batch_size = rotation6d.shape[0]
    pose6d = rotation6d.reshape(-1, 6)
    tmp_x = nn.functional.normalize(pose6d[:,:3], dim = -1)
    tmp_z = nn.functional.normalize(pose6d[:,3:], dim = -1)
    tmp_y = torch.cross(tmp_z, tmp_x, dim = -1)

    tmp_x = tmp_x.view(-1, 3, 1)
    tmp_y = tmp_y.view(-1, 3, 1)
    tmp_z = tmp_z.view(-1, 3, 1)
    R = torch.cat((tmp_x, tmp_y, tmp_z), -1)

    return R.reshape(batch_size, -1)

def rotation6d_2_rvecs(rotation6d):
    # Convert the 6D rotation representation to rotation matrices
    rot_mats = rotation6d_2_rot_mat(rotation6d).reshape((-1, 3, 3))
    rotation6d = rot_mats
    batch_size = rotation6d.shape[0]
    device = rotation6d.device
    dtype = rotation6d.dtype

    cos_theta = (torch.einsum('bii->b', rotation6d) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1, 1)

    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)

    r = rotation6d - rotation6d.transpose(1, 2)
    u = torch.stack([r[:, 2, 1], r[:, 0, 2], r[:, 1, 0]], dim=1)

    u_norm = F.normalize(u, dim=1)
    rvecs = u_norm * theta.view(-1, 1)

    # Handle the case where sin(theta) is close to 0
    small_angle_mask = sin_theta.abs() < 1e-5
    if small_angle_mask.any():
        r_small_angles = rotation6d[small_angle_mask]
        rvecs_small_angles = torch.atan2(r_small_angles[:, 2, 1], r_small_angles[:, 0, 0]) / 2
        rvecs[small_angle_mask] = rvecs_small_angles.unsqueeze(1)

    return rvecs


def rvecs_2_rotation6d(rvecs):
    # Convert the rotation vectors to rotation matrices
    numFrame, numJoint = rvecs.shape
    rot_mats = rvecs_2_rot_mat(rvecs)
    rot_mats = rot_mats.reshape((numFrame, numJoint // 3, 3, 3))

    # Extract the basis vectors from the rotation matrices
    tmp_x = rot_mats[..., :, 0]
    tmp_y = rot_mats[..., :, 1]
    tmp_z = rot_mats[..., :, 2]

    # Format the basis vectors into the 6D rotation representation
    rotation6d = torch.cat((tmp_x, tmp_z), -1)  # we use x and z as they were used in the original conversion to rotation matrix
    return rotation6d

