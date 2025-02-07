import numpy as np
import torch
import util.parser_util
import models.vqvae as vqvae
import torch.nn.functional as F
from util.vis_smpl import visualize_motion_in_smpl_global
from data_loaders.humanml.scripts.motion_process import recover_from_ric

from models.Transformer_Traj_Model_hml import TransformerAutoencoder_withCodes_hml_G2_Traj
import numpy as np
import util.traj_funcs as traj_funcs


def hardCode_inv_transform_traj(data):
    traj_std = torch.load( "./demo/info_traj_std.pt").to(data.device)
    traj_mean = torch.load( "./demo/info_traj_mean.pt").to(data.device)
    return data * traj_std + traj_mean

def hardCode_transform_traj(data):
    traj_std = torch.load( "./demo/info_traj_std.pt").to(data.device)
    traj_mean = torch.load( "./demo/info_traj_mean.pt").to(data.device)
    return (data - traj_mean) / traj_std

def hardCode_inv_transform(data):
    motion_std = torch.load( "./demo/info_motion_std.pt")
    motion_mean = torch.load( "./demo/info_motion_mean.pt")
    motion_std = torch.Tensor(motion_std).to(data.device)
    motion_mean = torch.Tensor(motion_mean).to(data.device)
    return data * motion_std + motion_mean

def create_fix_traj_in_vis(waypoints, L = 196):
    # convert trajectories into visulization mode
    # assert waypoints.shape == ( N(196), K(1), 3)
    # output shape (N, N*K, 3)
    N, K = waypoints.shape[:2]
    waypoints = waypoints.reshape((-1, 3))
    waypoints = waypoints.unsqueeze(0)
    waypoints = waypoints.repeat((L, 1, 1))
    return waypoints


def correct_foot_skid(foot_loc):

    HEIGHT_THRESHOLD = 0.02
    T, _, _ = foot_loc.shape
    corrected_foot_loc = foot_loc.clone()
    
    left_hight = torch.mean(foot_loc[:, [0, 1], :], dim=1)
    right_hight = torch.mean(foot_loc[:, [2, 3], :], dim=1)
    
    left_last_valid_loc = foot_loc[0, [0, 1], :]
    left_last_valid_loc[:, 2] = .0
    
    right_last_valid_loc = foot_loc[0, [2, 3], :]
    right_last_valid_loc[:, 2] = .0

    for t in range(T):
        if left_hight[t, 2] > HEIGHT_THRESHOLD:   
            left_last_valid_loc = foot_loc[t, [0, 1], :]
        else:
            corrected_foot_loc[t, [0, 1], :] = left_last_valid_loc
            corrected_foot_loc[t, [0, 1], 2] = .0
        
        if right_hight[t, 2] > HEIGHT_THRESHOLD:   
            right_last_valid_loc = foot_loc[t, [2, 3], :]
        else:
            corrected_foot_loc[t, [2, 3], :] = right_last_valid_loc
            corrected_foot_loc[t, [2, 3], 2] = .0 
    return corrected_foot_loc

def latent_fit(optimizer, smpl, source_kpts_model, static_vars, vp_model, extra_params={}, on_step=None, gstep=0, motionLen = 196, control_joints = [0], motion_traj_mask = None):

    data_loss = extra_params.get('data_loss', torch.nn.SmoothL1Loss(reduction='mean'))

    opt_map = [
        [0, 0], #root  
        [15, 1], #head
        [20, 2], #hand1    #left
        [21, 3], #hand2   #right
        [10, 4], #foot1   #left
        [11, 5], #foot2  #right
    ]
    opt_map = [opt_map[joint] for joint in control_joints]
    opt_jointNum = np.array(opt_map)[:, 0].tolist()
    opt_trajNum = np.array(opt_map)[:, 1].tolist()
    def fit(free_vars, motion_length, data_transform):
        fit.gstep += 1
        optimizer.zero_grad()
        
        pre_Joint = vp_model.vqvae.forward_decoder_from_quantized_codes(free_vars)
        
        sample = data_transform(pre_Joint[0].permute(1, 2, 0)).float()
        sample = recover_from_ric(sample, 22)[0, ...]
        sample[:, :, [1, 2]] = sample[:, :, [2, 1]]
        
        opt_objs = {}
        
        if motion_traj_mask is None:
            opt_objs['data'] = data_loss(sample[:motion_length, opt_jointNum,:], source_kpts_model['traj'][:motion_length,opt_trajNum,:].cuda()) #originally remove motion_length
        else:
            sample_masked = sample[:, [0, 15, 20, 21, 10, 11], :] * motion_traj_mask
            source_masked = source_kpts_model['traj'].cuda() * motion_traj_mask[:motion_length, ...]
            opt_objs['data'] = data_loss(sample_masked[:motion_length, opt_jointNum,:], source_masked[:motion_length,opt_trajNum,:])

        sample_foot_target = correct_foot_skid( sample[:motion_length, [7, 10, 8, 11], :])
        opt_objs['foot'] = 1.0 * data_loss(sample[:motion_length, [7, 10, 8, 11], :], sample_foot_target)
        
        loss_total = torch.sum(torch.stack(list(opt_objs.values())))
        loss_total.backward(retain_graph=True)
        fit.free_vars = free_vars
        fit.final_loss = loss_total
        return loss_total

    fit.gstep = gstep
    fit.final_loss = None
    fit.free_vars = {}
    return fit


def process(args, control_Joints, text_prompt, video_Length = 100):
    torch.cuda.set_device(args.gpu_idx)
    device = torch.device('cuda')
    weights_file = "./save_weights_vq/best_model_epoch_hml_emaReset.pth"   
    
    weights_file_transformer = "./save_weights/withEmaReset_stage3.pth" 
    
    # get transformer model
    transformer_traj_model = TransformerAutoencoder_withCodes_hml_G2_Traj(args).to(device)
    transformer_traj_model.load_state_dict(torch.load(weights_file_transformer), strict = False)
    transformer_traj_model.eval() 
    
    # get VQVAE model
    net = vqvae.HumanVQVAE(args,
                       args.num_emb,
                       args.emb_dim,
                       args.emb_dim).to(device)
    net.load_state_dict(torch.load(weights_file))
    net.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    frame_len = video_Length
    frame_with_padding = min(frame_len + 6, 196) # make the motion end in a smooth pose
    
    waypoints_traj = torch.zeros((196, 6, 3)).cuda()
    
    not_control_Joints = []
    for i in range(6):
        if i not in control_Joints:
            not_control_Joints.append(i)
    
    #####################
    # Draw Control Trajectories:
    
    x, y, z = traj_funcs. draw_curve_line((.0, 0.9, .0), 0.35, 2.5/frame_with_padding, theta_step = 30, steps=frame_with_padding)
    way_Pts_root = torch.Tensor(np.column_stack((x, y, z)).reshape((frame_with_padding, 1, -1))).cuda()
    waypoints_traj[:frame_with_padding, 0:1, :] = way_Pts_root #set traj for joint_0 i.e. root
    
    motion_traj_mask = None
    
    # Demo traj with mask
    '''
    motion_traj_mask = torch.ones_like(waypoints_traj).cuda()
    motion_traj_mask[40 :frame_with_padding - 20, 0:1, :] *= .0
    waypoints_traj = waypoints_traj *  motion_traj_mask
    '''
    ####################
    
    
    
    waypoints_traj = hardCode_transform_traj(waypoints_traj)
    
    waypoints_traj[:, not_control_Joints, :] *= .0  # mask the redundent traj information
    
    waypoints_traj = waypoints_traj.unsqueeze(0)
    
    with torch.no_grad():
        _, pre_codes = transformer_traj_model(waypoints_traj.to(device), text_prompt) # get inital codes
        
    
    
    #####################
    # Two ways of picking codes from distribution:
    codes_pick = torch.argmax(F.softmax(pre_codes, dim = -1), dim = -1)
    x_quantized_fromIds = net.vqvae.get_x_quantized_from_x_ids(codes_pick.permute(0, 2, 1))
    
    #codes_pick_gumbel_softmax = F.gumbel_softmax(pre_codes, tau=1, eps=1e-10, hard=True, dim = -1)
    #x_quantized_fromIds = net.vqvae.get_x_quantized_from_x_ids(codes_pick_gumbel_softmax.permute(0, 2, 3, 1).contiguous())
    #######################
    
    waypoints = waypoints_traj[0, :frame_len, :, :]
    waypoints = hardCode_inv_transform_traj(waypoints)
    waypoints[:, :, [1, 2]] = waypoints[:, :, [2, 1]]
    goal_dict = {"id": [0, 1, 2, 3, 4, 5],
                 "traj": waypoints[:, :,:] }
    
    # Prepare for Optimization
    x_quantized_init = x_quantized_fromIds 
    free_vars = []
    for ele in x_quantized_init:
        ele = ele.detach()
        ele.requires_grad = True
        free_vars.append(ele)
    
    optimizer = torch.optim.LBFGS(free_vars,
                                  lr=0.1,
                                  max_iter=1000,
                                  tolerance_change= 1e-6,#1e-10, #1e-30,
                                  max_eval=None,
                                  history_size= 200,
                                  line_search_fn='strong_wolfe')
    # Optimize
    gstep = 0
    closure = latent_fit(optimizer,
                     smpl = None,
                     source_kpts_model=goal_dict,
                     static_vars=None,
                     vp_model=net,
                     on_step=None,
                     gstep=gstep,
                     motionLen = frame_len,
                     control_joints=control_Joints,
                     motion_traj_mask = motion_traj_mask)
    optimizer.step(lambda: closure(free_vars, motion_length = frame_len, data_transform = hardCode_inv_transform))
    free_vars = closure.free_vars
    print("optimization done.")
    with torch.no_grad():
        output_all = net.vqvae.forward_decoder_from_quantized_codes(free_vars)
        output_all = output_all.cpu()
    
    pre_Joint = output_all.cpu()
    sample = hardCode_inv_transform(pre_Joint[0].permute(1, 2, 0)).float()
    sample = recover_from_ric(sample, 22)[0, ...]
    sample[:, :, [1, 2]] = sample[:, :, [2, 1]]
    trans = sample[:frame_len, 0, :]
    sample = sample[:frame_len, :, :]
    
    waypoints = create_fix_traj_in_vis(waypoints[:, control_Joints, :], frame_len)
    new_data_jointRot_tensor = torch.zeros((frame_len, 22, 3))
    visualize_motion_in_smpl_global(new_data_jointRot_tensor, None, None, trans, sample, None, waypoints)
    
if __name__ == "__main__":
    args = util.parser_util.mtm_args()
    control_joints = [0]  # 0->root 1->head 2->hand1 3->hand2 4->foot1 5->foot2; multi-joint control e.g. [0, 1, 3]
    process(args, control_joints, "a man raised both arms above head")
    #process(args, control_joints, "the person walks in circles")