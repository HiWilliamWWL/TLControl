import torch
from data_loaders.humanml.networks.modules import *
from data_loaders.humanml.networks.trainers import CompTrainerV6
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import os
def build_models(opt):
    if opt.text_enc_mod == 'bigru':
        text_encoder = TextEncoderBiGRU(word_size=opt.dim_word,
                                        pos_size=opt.dim_pos_ohot,
                                        hidden_size=opt.dim_text_hidden,
                                        device=opt.device)
        text_size = opt.dim_text_hidden * 2
    else:
        raise Exception("Text Encoder Mode not Recognized!!!")

    seq_prior = TextDecoder(text_size=text_size,
                            input_size=opt.dim_att_vec + opt.dim_movement_latent,
                            output_size=opt.dim_z,
                            hidden_size=opt.dim_pri_hidden,
                            n_layers=opt.n_layers_pri)


    seq_decoder = TextVAEDecoder(text_size=text_size,
                                 input_size=opt.dim_att_vec + opt.dim_z + opt.dim_movement_latent,
                                 output_size=opt.dim_movement_latent,
                                 hidden_size=opt.dim_dec_hidden,
                                 n_layers=opt.n_layers_dec)

    att_layer = AttLayer(query_dim=opt.dim_pos_hidden,
                         key_dim=text_size,
                         value_dim=opt.dim_att_vec)

    movement_enc = MovementConvEncoder(opt.dim_pose - 4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose)

    len_estimator = MotionLenEstimatorBiGRU(opt.dim_word, opt.dim_pos_ohot, 512, opt.num_classes)

    # latent_dis = LatentDis(input_size=opt.dim_z * 2)
    checkpoints = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_est_bigru', 'model', 'latest.tar'), map_location=opt.device)
    len_estimator.load_state_dict(checkpoints['estimator'])
    len_estimator.to(opt.device)
    len_estimator.eval()

    # return text_encoder, text_decoder, att_layer, vae_pri, vae_dec, vae_pos, motion_dis, movement_dis, latent_dis
    return text_encoder, seq_prior, seq_decoder, att_layer, movement_enc, movement_dec, len_estimator





class CompV6GeneratedDataset(Dataset):

    def __init__(self, opt, dataset, w_vectorizer, mm_num_samples, mm_num_repeats):
        assert mm_num_samples < len(dataset)
        print(opt.model_dir)

        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
        text_enc, seq_pri, seq_dec, att_layer, mov_enc, mov_dec, len_estimator = build_models(opt)
        trainer = CompTrainerV6(opt, text_enc, seq_pri, seq_dec, att_layer, mov_dec, mov_enc=mov_enc)
        epoch, it, sub_ep, schedule_len = trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))
        generated_motion = []
        mm_generated_motions = []
        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        min_mov_length = 10 if opt.dataset_name == 't2m' else 6
        # print(mm_idxs)

        print('Loading model: Epoch %03d Schedule_len %03d' % (epoch, schedule_len))
        trainer.eval_mode()
        trainer.to(opt.device)
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
                tokens = tokens[0].split('_')
                word_emb = word_emb.detach().to(opt.device).float()
                pos_ohot = pos_ohot.detach().to(opt.device).float()

                pred_dis = len_estimator(word_emb, pos_ohot, cap_lens)
                pred_dis = nn.Softmax(-1)(pred_dis).squeeze()

                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False

                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                for t in range(repeat_times):
                    mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)
                    if mov_length < min_mov_length:
                        mov_length = torch.multinomial(pred_dis, 1, replacement=True)

                    m_lens = mov_length * opt.unit_length
                    pred_motions, _, _ = trainer.generate(word_emb, pos_ohot, cap_lens, m_lens,
                                                          m_lens[0]//opt.unit_length, opt.dim_pose)
                    if t == 0:
                        # print(m_lens)
                        # print(text_data)
                        sub_dict = {'motion': pred_motions[0].cpu().numpy(),
                                    'length': m_lens[0].item(),
                                    'cap_len': cap_lens[0].item(),
                                    'caption': caption[0],
                                    'tokens': tokens}
                        generated_motion.append(sub_dict)

                    if is_mm:
                        mm_motions.append({
                            'motion': pred_motions[0].cpu().numpy(),
                            'length': m_lens[0].item()
                        })
                if is_mm:
                    mm_generated_motions.append({'caption': caption[0],
                                                 'tokens': tokens,
                                                 'cap_len': cap_lens[0].item(),
                                                 'mm_motions': mm_motions})

        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.opt = opt
        self.w_vectorizer = w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length < self.opt.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.opt.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)

class CompMDMGeneratedDataset(Dataset):

    def __init__(self, model, diffusion, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1.):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length
        sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )

        real_num_batches = len(dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        model.eval()


        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(dataloader)):
                
                print(f"\nCurrently Running Batch_{i}")

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                testShape = motion.shape
                                
                testShape = torch.Size((testShape[0], testShape[1], testShape[2], max_motion_length))
                
                for t in range(repeat_times):
                    print(f"Getting Sample Batch_{i} for the repeat time {t}, with current shape setting is {testShape}")
                    sample = sample_fn(
                        model,
                        testShape,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        noise=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )
                    print(f"Finish getting this sample: Batch_{i}_Repeat_{t}")
                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    } for bs_i in range(dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        } for bs_i in range(dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'caption': model_kwargs['y']['text'][bs_i],
                                    'tokens': tokens[bs_i],
                                    'cap_len': len(tokens[bs_i]),
                                    'mm_motions': mm_motions[bs_i::dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(dataloader.batch_size)]
                #break


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = dataloader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']

        if self.dataset.mode == 'eval':
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        '''
        print("START=======")
        print(word_embeddings.shape)
        print(pos_one_hots.shape)
        print(caption)
        print(sent_len)
        print(motion.shape)
        print(m_length)
        print(tokens)
        print("END=======")
        '''
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)
    
    
class CompMDMGeneratedDataset_Ours(Dataset):
    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['sent_len']
        traj = data['traj_data']

        if self.dataset.mode == 'eval':
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), traj
    
import pickle
def save_info_to_file(traj, text, filename='saved_data.pkl'):
    # Convert traj to a numpy array if it's not already one
    traj_array = traj.numpy() if not isinstance(traj, np.ndarray) else traj

    # Create a dictionary to store both the trajectory and the text
    data_to_save = {
        'trajectory': traj_array,
        'text': text
    }

    # Open the file in binary write mode and use pickle to serialize the data
    with open(filename, 'wb') as file:
        pickle.dump(data_to_save, file)

    print(f"Data saved to {filename}")

        

class CompMDMGeneratedDataset_MTM_Only(CompMDMGeneratedDataset_Ours):

    def __init__(self, transformer_traj_model, net, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, using_Root_Only = False):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length

        real_num_batches = len(dataloader)

        generated_motion = []
        
        with torch.no_grad():
            for i, (motion, model_kwargs) in enumerate(dataloader):

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]
                    
                traj_dataForm = torch.stack(model_kwargs['y']['traj_data'])
                if using_Root_Only:
                    mask_ids = [1, 2, 3, 4, 5]   #0-root, 1-head, 2-Lhand, 3-Rhand, 4-Lfoot, 5-Rfoot
                    traj_dataForm[:, :, mask_ids, :] *= .0
                output, pre_codes = transformer_traj_model(traj_dataForm.to(device), model_kwargs['y']['text'])
                codes_pick = torch.argmax(F.softmax(pre_codes, dim = -1), dim = -1)
                x_quantized_fromIds = net.vqvae.get_x_quantized_from_x_ids(codes_pick.permute(0, 2, 1).contiguous())
                sample = net.vqvae.forward_decoder_from_quantized_codes(x_quantized_fromIds)
        
                sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                            'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                            'caption': model_kwargs['y']['text'][bs_i],
                            'tokens': tokens[bs_i],
                            'sent_len': model_kwargs['y']['sent_len'][bs_i],
                            'traj_data': model_kwargs['y']['traj_data'][bs_i]
                            } for bs_i in range(dataloader.batch_size)]
                generated_motion += sub_dicts

        self.generated_motion = generated_motion
        self.mm_generated_motion = []
        self.w_vectorizer = dataloader.dataset.w_vectorizer
        
    

class CompMDMGeneratedDataset_MTM_Only_UpdateMax(CompMDMGeneratedDataset_Ours):

    def __init__(self, transformer_traj_model, net, dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, using_Root_Only = False):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert mm_num_samples < len(dataloader.dataset)
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        self.max_motion_length = max_motion_length

        real_num_batches = len(dataloader)

        generated_motion = []
        
        
        with torch.no_grad():
            for i, (motion, model_kwargs) in enumerate(dataloader):

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]
                    
                traj_dataForm = torch.stack(model_kwargs['y']['traj_data'])
                if using_Root_Only:
                    mask_ids = [1, 2, 3, 4, 5]   #0-root, 1-head, 2-Lhand, 3-Rhand, 4-Lfoot, 5-Rfoot
                    traj_dataForm[:, :, mask_ids, :] *= .0
                output, pre_codes = transformer_traj_model(traj_dataForm.to(device), model_kwargs['y']['text'])
                #codes_pick = torch.argmax(F.softmax(pre_codes, dim = -1), dim = -1)
                #x_quantized_fromIds = net.vqvae.get_x_quantized_from_x_ids(codes_pick.permute(0, 2, 1).contiguous())
                codes_pick_gumbel_softmax = F.gumbel_softmax(pre_codes, tau=1, eps=1e-10, hard=True, dim = -1) 
                x_quantized_fromIds = net.vqvae.get_x_quantized_from_x_ids(codes_pick_gumbel_softmax.permute(0, 2, 3, 1).contiguous())
                sample = net.vqvae.forward_decoder_from_quantized_codes(x_quantized_fromIds)
        
                sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                            'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                            'caption': model_kwargs['y']['text'][bs_i],
                            'tokens': tokens[bs_i],
                            'sent_len': model_kwargs['y']['sent_len'][bs_i],
                            'traj_data': model_kwargs['y']['traj_data'][bs_i]
                            } for bs_i in range(dataloader.batch_size)]
                generated_motion += sub_dicts

        self.generated_motion = generated_motion
        self.mm_generated_motion = []
        self.w_vectorizer = dataloader.dataset.w_vectorizer