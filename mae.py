import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

# supposed to be put in __init__.py of models folder
def get_model(name, **model_kwargs):
    return eval(name)(**model_kwargs)

class MAE(nn.Module):
    def __init__(self, image_channel, image_size, patch_size, enc_dim, dec_dim, encoder, decoder, mask_ratio=0.75) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * image_channel
        self.token_num = (image_size//patch_size)**2
        # Note that the input to the torch.nn.Transformer have the batch dimension in the middle: [T(token), B(batch), D(feature)]
        
        self.shuffler = CardShuffler(mask_ratio, self.token_num)
        
        self.register_buffer('enc_pos', positional_encoding(enc_dim, max_len=self.token_num))
        self.register_buffer('dec_pos', positional_encoding(dec_dim, max_len=self.token_num))
        
        self.mask_emb = nn.Parameter(torch.randn(dec_dim))
        
        self.in_proj = nn.Linear(self.patch_dim, enc_dim)
        self.encoder = Transformer(d_model=enc_dim, **encoder)
        self.mid_proj = nn.Linear(enc_dim, dec_dim) if enc_dim != dec_dim else nn.Identity()
        self.decoder = Transformer(d_model=dec_dim, **decoder)
        self.out_proj = nn.Linear(dec_dim, self.patch_dim) if dec_dim != self.patch_dim else nn.Identity()

        

    def forward(self, img, viz=False):
        
        self.shuffler.init_rand_idx(img.shape[0], img.device)
        
        
        patches = rearrange(img, 'b c (h s1) (w s2) -> (h w) b (s1 s2 c)', s1=self.patch_size, s2=self.patch_size)
        
        emb = self.in_proj(patches)

        _, enc_inp = self.shuffler.shuffle_split(emb + self.enc_pos)
        
        
        x = self.encoder(enc_inp)
        x = self.mid_proj(x)

        x = torch.cat([x, self.mask_emb.expand(self.token_num-x.shape[0], x.shape[1], -1)])
        dec_pos = self.shuffler.shuffle(self.dec_pos.expand_as(x))
        dec_out = self.decoder(x + dec_pos)

        pixel_recon = self.out_proj(dec_out)
        
        inpainted_patches, _ = self.shuffler.split(pixel_recon)
        
        # get target from input patches
        masked_patches, _ = self.shuffler.shuffle_split(patches)
        
        loss = F.mse_loss(inpainted_patches, masked_patches)
        
        if viz:
            img_recon = self.shuffler.unshuffle(pixel_recon)
            img_recon = rearrange('(h w) b (s1 s2 c) -> b c (h s1) (w s2)', h=img.shape[2]//self.patch_size, s1=self.patch_size, s2=self.patch_size)
            return {'loss':loss, 'recon': img_recon}
        
        return {'loss':loss}
        
class CardShuffler(nn.Module):
    def __init__(self, ratio=0.75, token_num=196):
        super().__init__()
        self.mask_n = int(ratio*token_num)
        self.token_n = token_num
        
    def init_rand_idx(self, batch_size, device) -> None:
        self.rand_idx = torch.rand(self.token_n, batch_size, device=device).argsort(dim=0)
        self.sort_idx = torch.argsort(self.rand_idx, dim=0).to(device)
        
    def shuffle(self, x):
        return x.gather(0, self.rand_idx.unsqueeze(-1).expand_as(x))
    
    def unshuffle(self, x):
        return x.gather(0, self.sort_idx.unsqueeze(-1).expand_as(x))
    
    def shuffle_split(self, x):
        x = self.shuffle(x)
        return x.split(self.mask_n)

    def split(self, x):
        return x.split(self.mask_n)
    
def positional_encoding(d_model, max_len=5000):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    return pe
    
class Transformer(nn.Module):
    def __init__(self, num_layers, norm, d_model, **layer_kwargs):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, **layer_kwargs),
            num_layers=num_layers,
            norm=norm
        )
    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)







if __name__ == '__main__':
    model = MAE(
        image_size=224,
        image_channel=3,
        patch_size=16,
        enc_dim=512,
        dec_dim=256,
        encoder=dict(
            num_layers=12,
            norm=None,
            nhead=8,
            dim_feedforward=2048,
            dropout=0,
            activation='relu'
        ),
        decoder=dict(
            num_layers=12,
            norm=None,
            # layer_kwargs=dict(
                nhead=4,
                dim_feedforward=1024,
                dropout=0,
                activation='relu'
            # )
        ),
        mask_ratio=0.75
    )
    x = torch.randn((2,3,224,224))
    y = model(x)
    y['loss'].backward()