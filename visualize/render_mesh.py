
import os
from visualize import vis_utils
from tqdm import tqdm

cases = ["omnicontrol_Test3/omi_Test3_5"]

if __name__ == '__main__':
    
    case_name = "jump_kick2" # omnicontrol_Test3/omi_Test3_2
    
    #for case_name in cases:
    npy_path = './test_case/' + case_name +'/state.pkl'
    results_dir = './test_case/'+case_name+'/SMPL_result/objs/'
    out_npy_path = './test_case/'+case_name+'/SMPL_result/SMPL_params'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    #npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i,
    #                            device=params.device, cuda=params.cuda)
    npy2obj = vis_utils.npy2obj(npy_path, 0, 0,
                                device=0, cuda=False)
    npy2obj.my_motion[:,0,:]
    npy2obj.joints[:,0,:]
    
    print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
    for frame_i in tqdm(range(npy2obj.real_num_frames)):
        npy2obj.save_obj(os.path.join(results_dir, 'frame{:03d}_v.obj'.format(frame_i)), frame_i)
        npy2obj.save_skel(os.path.join(results_dir, 'frame{:03d}_j.obj'.format(frame_i)), frame_i)
        npy2obj.save_motion(os.path.join(results_dir, 'frame{:03d}_m.obj'.format(frame_i)), frame_i)
    print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
    npy2obj.save_npy(out_npy_path)
