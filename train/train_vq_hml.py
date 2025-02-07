import os
import json
from util.tqdm_wrapper import tqdm
import torch
import torch.optim as optim
from data_loaders.get_data import get_dataset_loader
#from torch.utils.tensorboard import SummaryWriter
from torch import save
import models.vqvae as vqvae
import torch.nn.functional as F
import util.parser_util
import warnings
warnings.filterwarnings('ignore')

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

device = torch.device('cuda')
##### ---- Exp dirs ---- #####
args = util.parser_util.mtm_args()
torch.manual_seed(args.seed)



##### ---- Dataloader ---- #####
data = get_dataset_loader(name="humanml", batch_size=args.batch_size, num_frames=196, split="train")

##### ---- Network ---- #####
net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.num_emb,
                       args.emb_dim,
                       args.output_emb_width)


if len(args.resume_pth) > 0 : 
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
net.train()
#net.cuda()
net.to(device)

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr = args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
#optimizer = optim.AdamW(net.parameters(), lr = args.lr, betas=(0.9, 0.99))
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)


##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

num_epochs = 1000
num_warm_up_iter = 100
num_print_iter = 10

##### ---- Training ---- #####
min_reconError = 1e7
for nb_iter in range(1, num_epochs + 1):
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
    print(f"Start training Epoch {nb_iter}...")
    progress_bar = tqdm(enumerate(data), total=len(data), desc="Epoch {:03d}".format(nb_iter))
    for i, batch in progress_bar:
        optimizer.zero_grad()  # Reset the gradients
        motion, cond = batch
        motion = motion.cuda()
    
        pred_motion, loss_commit, perplexity = net(motion)
        loss_motion = F.mse_loss(pred_motion, motion)
        loss = loss_motion + args.commit * loss_commit
        
        loss.backward()
        optimizer.step()
        #scheduler.step()
        
        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += args.commit * loss_commit.item()
    avg_recons /= len(data)
    avg_perplexity /= len(data)
    avg_commit /= len(data)
        
    print(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
    
    if nb_iter % num_print_iter ==  0 :
        print("Check if saave...?")
        if avg_recons < min_reconError:
            save(net.state_dict(), f"./save_weights_vq/best_model_epoch_hml_{avg_recons}AVG.pth")
            print("----saved----")
            min_reconError = avg_recons
        