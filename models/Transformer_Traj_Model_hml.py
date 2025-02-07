import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import random

'''
def random_mask(x, mask_rates, increase_factor=0.5):
    mask = torch.ones_like(x)
    time_steps = x.shape[0]
    
    for i, mask_rate in enumerate(mask_rates):
        for t in range(time_steps):
            if t > 0 and mask[t-1, i, 0] == 0:
                # If the previous frame is masked, increase the mask rate
                if torch.rand(1) < mask_rate + increase_factor:
                    mask[t, i, :] = 0
            elif torch.rand(1) < mask_rate:
                mask[t, i, :] = 0
    return x * mask
'''

rootPart = [0, 3, 6, 9, 12, 15]
armPart1 = [14, 17, 19, 21]
armPart2 = [13, 16, 18, 20]
legPart1 = [2, 5, 8, 11]
legPart2 = [1, 4, 7, 10]
partSeg = [rootPart, armPart1, armPart2, legPart1, legPart2]

def random_mask(x, mask_rates):
    x_using = x.clone()
    mask = torch.ones_like(x_using[:, :, :, 0])
    for i, mask_rate in enumerate(mask_rates):
        mask[:, :, i].bernoulli_(1-mask_rate)
    mask = mask.unsqueeze(-1)
    mask = mask.repeat(1, 1, 1, 3)
    x_ones = torch.ones_like(x_using)
    #x_inf = x_ones * 5.0
    return x_using * mask  + (x_ones - mask) * -50.0


def random_mask_seq(x, mask_rates, max_mask_len = 50, joint_mask = None):
    x_using = x.clone()
    T = x_using.size(1)
    data_dim = x_using.size(-1)
    
    mask = torch.ones_like(x_using[:, :, :, 0])
    mask_joints = None
    rand_number = random.random()
    
    if rand_number < 0.2:
        mask *= .0
    else:
        if joint_mask is not None and rand_number < 0.7:
            mask_joints = random.sample([0,1,2,3,4,5], joint_mask)
        for i, mask_rate in enumerate(mask_rates):
            if mask_joints is not None and i in mask_joints:
                mask[:, :, i] *= .0
                continue
            total_masked = 0
            need_masked = int(round(mask_rate * T))
            while total_masked < need_masked:
                center = torch.randint(0, T, (1,)).item()
                if total_masked < need_masked - max_mask_len:
                    length = torch.randint(1, max_mask_len+1, (1,)).item()
                else:
                    length = need_masked - total_masked
                    
                left = max(0, center - length // 2)
                right = min(T, left + length)

                mask[:, left:right, i] *= .0
                total_masked = int(T - torch.sum(mask[0, :, i]).item())
    mask = mask.unsqueeze(-1)
    mask = mask.repeat(1, 1, 1, data_dim)
    return x_using * mask#+ (x_ones - mask) * -50.0


def random_mask_seq_update(x, mask_rates, max_mask_len = 15, joint_mask = 5):
    x_using = x.clone()
    T = x_using.size(1)
    data_dim = x_using.size(-1)
    
    mask = torch.ones_like(x_using[:, :, :, 0])
    mask_joints = None
    rand_number = random.random()
    
    if rand_number < 0.1:
        return x_using

    if joint_mask is not None and rand_number < 0.8:
        mask_joints = random.sample([0,1,2,3,4,5], 5)
        mask[:, :, mask_joints] *= .0
    else:
        for i, mask_rate in enumerate(mask_rates):
            total_masked = 0
            need_masked = int(round(mask_rate * T))
            while total_masked < need_masked:
                center = torch.randint(0, T, (1,)).item()
                if total_masked < need_masked - max_mask_len:
                    length = torch.randint(1, max_mask_len+1, (1,)).item()
                else:
                    length = need_masked - total_masked
                    
                left = max(0, center - length // 2)
                right = min(T, left + length)

                mask[:, left:right, i] *= .0
                total_masked = int(T - torch.sum(mask[0, :, i]).item())
    mask = mask.unsqueeze(-1)
    mask = mask.repeat(1, 1, 1, data_dim)
    return x_using * mask




class TransformerAutoencoder_hml(nn.Module):
    def __init__(self, args):
        super(TransformerAutoencoder_hml, self).__init__()
        
    
    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
    
    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        #device = next(self.parameters()).device
        device = "cuda"
        max_text_len = 20  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            #print('texts', texts.shape)
            
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            #print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()


class TransformerAutoencoder_withCodes_hml_G2_Traj(TransformerAutoencoder_hml):
    def __init__(self, args):
        super(TransformerAutoencoder_withCodes_hml_G2_Traj, self).__init__(args)
        self.input_dim = args.input_dim
        #if self.input_dim > 3:
        #    self.input_dim *= 28
        self.codesTimeLen = 49
        self.codes_realLength = 196 // self.codesTimeLen  #4
        
        self.model_dim1 = 512
        self.model_dim2 = 256
        
        self.clip_dim = 512 #clip_dim
        
        self.clip_version = 'ViT-B/32'
        self.clip_model = self.load_and_freeze_clip(self.clip_version)
        self.embed_text = nn.Linear(self.clip_dim, self.model_dim1)
        
        
        self.linear_in = nn.Linear(self.codes_realLength * self.input_dim, self.model_dim1)
        
        #self.pos_encoder = PositionalEncoding(self.model_dim_part, args.dropout)
        self.pos_encoder = PositionalEncoding(self.model_dim1, args.dropout)
        
        #self.tokenEmb = TokenTypeEncoding(self.model_dim, self.pos_encoder)
        
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.model_dim1,
            nhead= 4, #args.num_heads,
            dim_feedforward=self.model_dim1 * 4,
            dropout=args.dropout)
        # Define transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=4)
        
        self.linear_mid_codeIdx = nn.Linear(self.model_dim1, self.model_dim2)
        #self.linear_mid_traj = nn.Linear(self.model_dim1, self.model_dim2)
        
        self.num_emb = args.num_emb
        
        decoder_layers1 = nn.TransformerEncoderLayer(
            d_model=self.model_dim2,
            nhead=4,  # Number of heads
            dim_feedforward=self.model_dim2 * 4,
            dropout=args.dropout
            )
            # Define transformer decoder
        self.transformer_codeIdx = nn.TransformerEncoder(
                encoder_layer=decoder_layers1,
                num_layers=3
            )
        self.project_codeIdx = nn.Linear(self.model_dim2, self.num_emb )
        
    
    def forward(self, x, x_text): #Toekn size: T
        # x: bs, T, NumJoint, 3
        # y: bs, t, NumJoint
        #x = x.squeeze()
        enc_text = self.encode_text(x_text)
        text_feature = self.embed_text(enc_text).unsqueeze(0)
        #x = x.reshape((-1, 196, 6*3))
        bs, T, NJoints, input_dim = x.shape
        x = x.permute(2, 1, 0, 3) # x:  NumJoint, T, bs, model_dim
        Tt = T // self.codes_realLength
        x = x.reshape(NJoints, Tt, self.codes_realLength, bs, input_dim)
        x = x.permute(1, 0, 3, 2, 4)  
        x = x.reshape(Tt * NJoints, bs, self.codes_realLength * input_dim)   
        
        x = self.linear_in(x) # x: bs, T, NumJointm, model_dim
        
        x = torch.cat([text_feature, x], dim = 0)
        
        x = self.pos_encoder(x)
        
        result_text = self.transformer_encoder(x)
        #result = result_text[1:, :, :] # Tt*NJoints, bs, model_dim
        
        
        pre_codes = self.transformer_codeIdx(self.linear_mid_codeIdx(result_text))
        pre_codes = pre_codes[1:, :, :]
        pre_codes = self.project_codeIdx(pre_codes)
        pre_codes = pre_codes.reshape(Tt, NJoints, bs, -1)
        pre_codes = pre_codes.permute(2, 1, 0, 3)  
        return None, pre_codes
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)

class TokenTypeEncoding(nn.Module):
    def __init__(self, d_model, sequence_pos_encoder):
        super().__init__()
        self.d_model = d_model
        self.sequence_pos_encoder = sequence_pos_encoder
        self.lineEmb = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).cuda()
        #self.lineEmb28 = torch.torch.linspace(0, 1, steps=28).cuda()

    def forward(self, x, shapeInfo):
        NJoints, Tbs, model_dim = shapeInfo
        for i in range(NJoints):
            x[i, :, :] += self.lineEmb[i]
            #x[i, :, :] += self.lineEmb28[i]
        return x

class TokenEncoding(nn.Module):
    def __init__(self, num_tokens, d_model):
        super(TokenEncoding, self).__init__()
        self.token_embeddings = nn.Embedding(num_tokens, d_model)

    def forward(self, x):
        # x: The input tensor of shape (N, ..., D)
        token_ids = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        token_embeds = self.token_embeddings(token_ids)

        # Reshape token_embeds for broadcasting
        # Add extra dimensions (1s) after the first dimension to match the shape of x
        shape = [x.size(0)] + [1] * (x.dim() - 2) + [x.size(-1)]
        token_embeds = token_embeds.view(*shape)
        return x + token_embeds