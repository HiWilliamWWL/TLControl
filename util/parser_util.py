from argparse import ArgumentParser
import argparse
import os
import json


def add_transformer_traj_model_options(parser):
    group = parser.add_argument_group('transformer')
    parser.add_argument("--model_dim", default=128, type=int, help="dimention of the model")
    parser.add_argument("--num_heads", default=4, type=int, help="number of heads")
    parser.add_argument("--num_layers", default=4, type=int, help="number of layers")
    parser.add_argument("--clip_dim", default=512, type=int, help="dimention of the CLIP")
    parser.add_argument("--block_size", default=49, type=int, help="dimention of the blocks")
    parser.add_argument("--ff_rate", default=4, type=int, help="FF forward Multiple")
    
    parser.add_argument("--input_dim", default=3, type=int, help="3 = 3d xyz positions; or 6 = 3d-xyz + 0,0,0; or 6d-rotation")
    parser.add_argument("--load_dir", default="./save_weights/best_model_epoch.pth", type=str, help="weight file")

def add_vqvae_model_options(parser):
    group = parser.add_argument_group('vqvae')
    parser.add_argument("--hidden_dim", default=128, type=int, help="dimention of the model")
    parser.add_argument("--emb_dim", default=128, type=int, help="embedding dimension")
    parser.add_argument("--num_emb", default=128, type=int, help="number of embeddings")
    parser.add_argument("--output_emb_width", default=128, type=int, help="output_emb_width 512")
    
    parser.add_argument("--traj_dim", default=64, type=int, help="dimension of trajectory feature")
    parser.add_argument("--load_dir_vqvae", default="./save_weights_vq/best_model_epoch_hml_emaReset.pth", type=str, help="weight file")
    parser.add_argument("--out_dir", default="./save_weights_vq/", type=str, help="weight file")
    parser.add_argument("--resume_pth", default="", type=str, help="weight file")
    parser.add_argument("--quantizer", default="ema_reset", type=str, help="quantizer type   ema_reset   reset  orig")
    parser.add_argument("--commit", type=float, default=0.001, help="hyper-parameter for the commitment loss")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")

def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    parser.add_argument("--data_rot_dim", default=6, type=int, help="use 6d rotation in the data")
    parser.add_argument('--amass_root', type=str, default='./data/AMASS/', help='Root directory of AMASS dataset.')
    parser.add_argument('--datasets', type=str, nargs='+', default=None, help='Which datasets to process. By default processes all.')
    parser.add_argument('--output_path', type=str, default='./output', help='Root directory to save output to.')
    parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--gpu_idx", type=int, default=0, help="select gpu")
    parser.add_argument('--train', dest="train", action="store_true", help='train or test')
    parser.add_argument("--resume_checkpoint", default=None, type=str, help="Path to specific checkpoint for resume training.")
    parser.add_argument('--visual', dest="visual", default=False, action="store_true", help='visualize mesh')
    parser.add_argument("--loss_weight", default=1, type=int, help="loss weight")
    
    parser.add_argument("--dropout", default=0.1, type=int, help="dropout")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate") #0.001
    parser.add_argument("--seed", default=8, type=int, help="dropout")
    
    parser.add_argument("--output_dim", default=6, type=int, help="loss weight")
    parser.add_argument('--load_data_fromPack', action="store_true", help='load data')
    parser.add_argument('--max_len', default=100, type=int, help="the maximum length of a sequence during training")



def mtm_args():
    parser = ArgumentParser()
    add_data_options(parser)
    add_transformer_traj_model_options(parser)
    add_vqvae_model_options(parser)
    return parser.parse_args()