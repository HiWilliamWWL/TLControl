import torch.nn as nn
import torch
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset

#import numpy as np
class VQVAE_limb_hml(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width= 512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        #self.partSeg = [[0], [3, 6, 9, 12, 15], [14, 17, 19, 21], [13, 16, 18, 20], [2, 5, 8, 11], [1, 4, 7, 10]]
        def values_term(i):
            i -= 1
            return [4+i*3, 4+i*3+1, 4+i*3+2] + [4 + 63+i*6 + k for k in range(6)] + [4+63+126+(i+1)*3 + k for k in range(3)]
        self.partSeg = [[0, 1, 2, 3, 4+63+126, 4+63+126+1, 4+63+126+2],
                        [x for i in [3, 6, 9, 12, 15] for x in values_term(i)],
                        [x for i in [13, 16, 18, 20] for x in values_term(i)],
                        [x for i in [14, 17, 19, 21] for x in values_term(i)],
                        [x for i in [1, 4, 7, 10] for x in values_term(i)] + [259, 260],
                        [x for i in [2, 5, 8, 11] for x in values_term(i)] + [261, 262]]
        #single_limb_emb_width = output_emb_width // len(self.partSeg)
        #self.encoder = Encoder(22 * args.data_rot_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(263, output_emb_width * len(self.partSeg), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        if args.quantizer == "ema_reset":
            self.quantizers = nn.ModuleList([QuantizeEMAReset(nb_code, code_dim, args) for part in self.partSeg])
        elif args.quantizer == "orig":
            self.quantizers = nn.ModuleList([Quantizer(nb_code, code_dim, 1.0) for part in self.partSeg]) 
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizers = nn.ModuleList([QuantizeReset(nb_code, code_dim, args) for part in self.partSeg])
        self.limb_encoders = nn.ModuleList([
            Encoder(len(part), output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
            for part in self.partSeg
        ])
        
        
    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x

    def forward(self, x):
        
        #x_in = self.preprocess(x)
        # Encode
        #x_s = [self.limb_encoders[i](x_in[:, part, :]) for i, part in enumerate(self.partSeg)]
        
        x = x.squeeze()
        bs, F, T = x.shape
        x_in = x.permute(0, 2, 1)
        x_s = []
        
        for i, part in enumerate(self.partSeg):
            x_current = x_in[:, :, part].reshape(bs, T, -1).cuda()
            x_current = self.preprocess(x_current)
            x_feature = self.limb_encoders[i](x_current)
            
            #x_feature.shape => (Batch, 128, 49)
            #torch.norm(x_feature, dim=[1, 2]).shape => (Batch, 1, 1)
            x_feature = x_feature / torch.norm(x_feature, dim=[1, 2]).unsqueeze(1).unsqueeze(1)
            
            x_s.append(x_feature)
            
        x_quantized = []
        loss = .0
        perplexity = .0
        for i, part in enumerate(self.partSeg):
            x, l, p  = self.quantizers[i](x_s[i])
            x_quantized.append(x)
            loss += l
            perplexity += p
        x_quantized = torch.cat(x_quantized, dim = 1)
        

        ## decoder
        x_decoder = self.decoder(x_quantized)
        
        x_out = x_decoder.unsqueeze(2)
        #x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity
    
    def get_quantized_codes(self, x, in_idx_format = True):
        x = x.squeeze(-2)
        bs, F, T = x.shape
        x_in = x.permute(0, 2, 1)
        x_s = []
        for i, part in enumerate(self.partSeg):
            x_current = x_in[:, :, part].reshape(bs, T, -1).cuda()
            x_current = self.preprocess(x_current)
            x_feature = self.limb_encoders[i](x_current)
            
            x_feature = x_feature / torch.norm(x_feature, dim=[1, 2]).unsqueeze(1).unsqueeze(1) 
            
            x_s.append(x_feature)
        if not in_idx_format:
            x_quantized = []
            for i, part in enumerate(self.partSeg):
                x, l, p  = self.quantizers[i](x_s[i])
                x_quantized.append(x)
            return x_quantized
        else:
            x_ids = []
            for i, part in enumerate(self.partSeg):
                x_ids.append(self.quantizers[i].get_code_idx(x_s[i]).unsqueeze(-2))
            x_ids = torch.cat(x_ids, dim = 1)
            return x_ids
    
    def forward_decoder(self, x_id):
        # x.shape = ()
        x_quantized = []
        for i, part in enumerate(self.partSeg):
            x_current = self.quantizers[i].forward_from_code_idx(x_id[:, :, i])
            x_quantized.append(x_current)
        x_quantized = torch.cat(x_quantized, dim = 1)

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out
    
    def get_x_quantized_from_x_ids(self, x_id):
        x_quantized = []
        #print(f"inside   {x_id.shape}")
        for i, part in enumerate(self.partSeg):
            x_current = self.quantizers[i].forward_from_code_idx(x_id[..., i])
            x_quantized.append(x_current)
        return x_quantized

    def forward_decoder_from_quantized_codes(self, x_quantized):
        x_quantized = torch.cat(x_quantized, dim = 1)
        x_decoder = self.decoder(x_quantized)
        x_out = x_decoder.unsqueeze(2)
        return x_out

    
class HumanVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        
        
        self.nb_joints = 22
        self.vqvae = VQVAE_limb_hml(args, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x):

        x_out, loss, perplexity = self.vqvae(x)
        
        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
    
    def get_code_idx(self, x):
        return self.vqvae.get_quantized_codes(x)