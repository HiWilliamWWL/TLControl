import os
from util.tqdm_wrapper import tqdm
import torch
from data_loaders.get_data import get_dataset_loader
from torch.utils.tensorboard import SummaryWriter
from torch import save
import models.vqvae as vqvae
import torch.nn.functional as F
import util.parser_util
#from data.data_mtm_with_joints import process_traj_data
from models.Transformer_Traj_Model_hml import TransformerAutoencoder_withCodes_hml_G2_Traj, random_mask_seq_update
from torch.optim import AdamW
import models.vqvae as vqvae
import math
from testing_eval import eval_humanml_mtm
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper

device = torch.device('cuda')
##### ---- Exp dirs ---- #####
args = util.parser_util.mtm_args()
torch.manual_seed(args.seed)



data = get_dataset_loader(name="humanml", batch_size=args.batch_size, num_frames=196, hml_mode = "train", split="train", args = None)


data_eval_dataset = get_dataset_loader(name="humanml", batch_size=32, num_frames=196, hml_mode = "eval", split="val", args = None)
data_eval_dataset_gt = get_dataset_loader(name="humanml", batch_size=32, num_frames=196, hml_mode = "gt", split="val", args = None)

MIN_LR = .0000001
FIX_LR_TIME = 500
def update_lr(optimizer, nb_iter, num_epochs, warm_up_iter, lr):
    if nb_iter < FIX_LR_TIME:
        return optimizer, lr
    lr *= 0.5
    if nb_iter < warm_up_iter:
        current_lr = lr * (nb_iter+1)  / warm_up_iter
    else:
        current_lr = lr * ( (num_epochs - nb_iter)  / (num_epochs - warm_up_iter))
        current_lr = MIN_LR if current_lr < MIN_LR else current_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

vq_net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.num_emb,
                       args.emb_dim,
                       args.output_emb_width)
vq_net.load_state_dict(torch.load(args.load_dir_vqvae))
vq_net.eval()  # Set the model to evaluation mode
vq_net.to(device)
#for i in range(6):
#    vq_net.vqvae.quantizers[i].codebook.requires_grad = True


reload_transformer_path = None
transformer_traj_model = TransformerAutoencoder_withCodes_hml_G2_Traj(args).to(device)  #TransformerAutoencoder_withCodes  TransformerAutoencoder_withCodes_simplified
if reload_transformer_path is not None:
    transformer_traj_model.load_state_dict(torch.load(reload_transformer_path))
    print(f"Loaded Transformer Weights from {reload_transformer_path}")

optimizer = AdamW(transformer_traj_model.parameters(), lr=args.lr, betas=(0.9, 0.99))
writer = SummaryWriter()
torch.manual_seed(args.seed)

# Mask rates for each of the 6 data
# head, l_hand, r_hand, root, l_ankle, r_ankle
mask_rates0 = [.0, 0., 0., 0., 0., 0.]
mask_rates1 = [.2, 0.2, 0.2, 0.2, 0.2, 0.2]
mask_rates2 = [.4, 0.4, 0.4, 0.4, 0.4, 0.4]
mask_rates3 = [.5, 0.5, 0.5, 0.5, 0.5, 0.5]
mask_rates = [mask_rates0, mask_rates1, mask_rates2, mask_rates3]

num_epochs = 2000  # specify your number of epochs here
warm_up_iter = 0
epoch_gap = [0, 200, 250, 300]
evaluate_gap = 30
best_loss = [1e7, 1e7, 1e7, 1e7]
best_acc = [.0, .0, .0, .0]
best_training_loss = 1e7
best_FID1 = 1e7
best_FID2 = 1e7

best_eval = float('inf')  # Keep track of the best loss
bce_loss_fn = torch.nn.CrossEntropyLoss()

weight_dir = "./save_weights/"

eval_data = {
    'test_fullJoints': lambda: eval_humanml_mtm.get_mtm_loader(
        transformer_traj_model, vq_net, 32,
        data_eval_dataset, 0, 0, 196, 1000, False),
    'test_rootOnly': lambda: eval_humanml_mtm.get_mtm_loader(
        transformer_traj_model, vq_net, 32,
        data_eval_dataset, 0, 0, 196, 1000, True)
}
eval_wrapper = EvaluatorMDMWrapper("humanml", torch.device("cuda"))


for epoch in range(num_epochs):  # Loop over the dataset multiple times
    transformer_traj_model.train()  # Set the model to training mode
    running_loss = 0.0  # To keep track of the loss over an epoch
    optimizer, current_lr = update_lr(optimizer, epoch, num_epochs, warm_up_iter, args.lr)
    print(f"Start training Epoch {epoch+1}...with lr: {current_lr}")
    progress_bar = tqdm(enumerate(data), total=len(data), desc="Epoch {:03d}".format(epoch+1))
    mask_index = next((i for i, v in enumerate(epoch_gap) if epoch < v), len(epoch_gap) - 1)
    traj_loss_sum = .0
    for i, batch in progress_bar:

        optimizer.zero_grad()  # Reset the gradients
        
        motion, cond = batch
        batch_size = motion.shape[0]
        
        x_label_idx = vq_net.get_code_idx(motion.to(device))#.permute(0, 2, 1)
        x_label_idx_oneHot = F.one_hot(x_label_idx, num_classes=args.num_emb).long() #torch.Size([Batch_size, 6, 49, 128])
        
        traj_info = torch.cat(cond['y']['traj_data'], dim=0).reshape((-1, 196, 6, 3))
        #print(motion.shape)  # B, 263, 1, 196
        #print(traj_info.shape)  # B, 196, 6, 3
        #print(x_label_idx.shape) #torch.Size([64, 6, 49])
        
        lengths = cond['y']['lengths']
        
        traj_info = traj_info.to(device) 
        #x_joint = x_joint.to(device)
        
        joint_mask = 5
        x_traj_masked = random_mask_seq_update(traj_info, mask_rates[mask_index], joint_mask = joint_mask)
        
        _, pre_codes = transformer_traj_model(x_traj_masked.to(device), cond['y']['text'])
        
        #codes_pick = torch.argmax(F.softmax(pre_codes, dim = -1), dim = -1)
        codes_pick_gumbel_softmax = F.gumbel_softmax(pre_codes, tau=1, eps=1e-10, hard=True, dim = -1)
        
        
        x_quantized_fromIds = vq_net.vqvae.get_x_quantized_from_x_ids(codes_pick_gumbel_softmax.permute(0, 2, 3, 1).contiguous())
        sample = vq_net.vqvae.forward_decoder_from_quantized_codes(x_quantized_fromIds)
        
        reshaped_pre_codes = pre_codes.permute(0, 3, 1, 2)
        
        loss = .0
        for i in range(batch_size):
            current_len = math.ceil(cond['y']['lengths'][i] / 4)
            
            bce_loss = bce_loss_fn(reshaped_pre_codes[i:i+1, :, :, :current_len], x_label_idx[i:i+1, :, :current_len]) / batch_size
            
            loss += bce_loss #+ traj_loss
            
        loss += 0.001 * F.mse_loss(sample, motion.cuda())
        
        loss.backward()  # Compute the gradients
        optimizer.step()  # Update the weights
        
        running_loss += loss.item()
        progress_bar.set_postfix({'loss': running_loss / (i+1)})  # update the progress bar with the current loss
        

    # Print average loss over the epoch
    training_loss = running_loss/len(data)
    
    print(f"Epoch {epoch+1}, Epoach Training Loss: {training_loss};")
    writer.add_scalar('Loss/train', training_loss, epoch)
    
    if training_loss < best_training_loss:
        best_training_loss = training_loss
        weight_file = os.path.join(weight_dir, f'withEmaReset_BEST_Loss_END.pth')
        save(transformer_traj_model.state_dict(), weight_file)
    
    # Evaluate the model on the evaluation set every evaluate_gap epochs
    if epoch >= epoch_gap[1]  and (epoch % evaluate_gap == 0) :
    #if epoch % evaluate_gap == 0:
        transformer_traj_model.eval()  # Set the model to evaluation mode
        total_num = 0.0
        acc_num = 0.0
        
        with torch.no_grad():  # Do not compute gradients in this block
            log_file = os.path.join(weight_dir, f'eval_humanml_{epoch}.log')
            diversity_times = 300
            mm_num_times = 0  # mm is super slow hence we won't run it during training
            eval_dict = eval_humanml_mtm.evaluation(
                eval_wrapper, data_eval_dataset_gt, eval_data, log_file,
                replication_times=1, diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=False)

            filtered_dict = {key: value for key, value in eval_dict.items() if "ground" not in key}
            #wandb.log(filtered_dict)
            current_FID1 = eval_dict['FID_test_fullJoints']
            current_FID2 = eval_dict['FID_test_rootOnly']
            
            # Save the model if it has the best evaluation loss so far
            if current_FID1 < best_FID1 or current_FID2 < best_FID2:
                best_FID1 = min(best_FID1, current_FID1)
                best_FID2 = min(best_FID2, current_FID2)
                weight_file = os.path.join(weight_dir, f'trans_stage{mask_index}_FIDfull{best_FID2}_END.pth')
                save(transformer_traj_model.state_dict(), weight_file)
                print(f"------New best model saved with Evaluation FIDfull: {best_FID1}-------")
print('Finished Training')
writer.close()