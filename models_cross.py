# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import numpy as np
import math
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from util.visualize import visualize_noise_image
from einops import rearrange

from transformer_utils import Block, CrossAttentionBlock, PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed
from diffusion_utils import TimestepEmbedder

class WeightedFeatureMaps(nn.Module):
    def __init__(self, k, embed_dim, *, norm_layer=nn.LayerNorm, decoder_depth):
        super(WeightedFeatureMaps, self).__init__()
        self.linear = nn.Linear(k, decoder_depth, bias=False)
        
        std_dev = 1. / math.sqrt(k)
        nn.init.normal_(self.linear.weight, mean=0., std=std_dev)

    def forward(self, feature_maps):
        # Ensure the input is a list
        assert isinstance(feature_maps, list), "Input should be a list of feature maps"
        # Ensure the list has the same length as the number of weights
        assert len(feature_maps) == (self.linear.weight.shape[1]), "Number of feature maps and weights should match"
        stacked_feature_maps = torch.stack(feature_maps, dim=-1)  # shape: (B, L, C, k)
        # compute a weighted average of the feature maps
        # decoder_depth is denoted as j
        output = self.linear(stacked_feature_maps)
        return output

class MaskedAutoencoderViT(nn.Module):
    """ 
    Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 weight_fm=False, 
                 use_fm=[-1], use_input=False, self_attn=False
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim) # these are needed regardless of the patch sampling method
        self.decoder_patch_embed = PatchEmbed(img_size, patch_size, in_chans, decoder_embed_dim) # these are needed regardless of the patch sampling method
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        # --------------------------------------------------------------------------
        # weighted feature maps for cross attention
        self.weight_fm = weight_fm
        self.use_input = use_input # use input as one of the feature maps
        if len(use_fm) == 1 and use_fm[0] == -1:
            self.use_fm = list(range(depth))
        else:
            self.use_fm = [i if i >= 0 else depth + i for i in use_fm]
        if self.weight_fm:
            dec_norms = []
            for i in range(decoder_depth):
                norm_layer_i = norm_layer(embed_dim)
                dec_norms.append(norm_layer_i)
            self.dec_norms = nn.ModuleList(dec_norms)

            # feature weighting
            self.wfm = WeightedFeatureMaps(len(self.use_fm) + (1 if self.use_input else 0), embed_dim, norm_layer=norm_layer, decoder_depth=decoder_depth)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        print("use self attention: ", self_attn)
        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, self_attn=self_attn)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------
        # Dealing with positional embedding, patch sampling 
        # encoder
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # decoder 
        
        # --------------------------------------------------------------------------
        # NOTE : Diffusion utils
        self.diffusion_timesteps = 1  # Number of Diffusion Steps during training. Note, this is much smaller than ordinary (usually ~1000)
        beta_start = 0.02 # Default from paper
        beta_end = 0.02  # NOTE: This is different to accomodate the much shorter DIFFUSION_TIMESTEPS (Usually ~1000). For 1000 diffusion timesteps, use 0.02.
        clip_sample = False  # Used for better sample generation
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.diffusion_timesteps, beta_schedule="linear", beta_start=beta_start, beta_end=beta_end, clip_sample=clip_sample)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # timestep embedding
        self.t_embedder = TimestepEmbedder(decoder_embed_dim)
        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, kept_mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio)) # visible patch
        len_masked = int(L * (mask_ratio - kept_mask_ratio)) # masked patch
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] # visible patch
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # get_masked_patch

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :(len_keep + len_masked)] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def grid_patchify(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        return x
    
    def decoder_grid_patchify(self, x, t):
        # embed patches
        x = self.decoder_patch_embed(x)
        t = self.t_embedder(t)
        c = self.decoder_pos_embed[:, 1:, :] + t
        # add pos embed w/o cls token
        x = x + c
        return x
    
    def mask_tokens_grid(self, noise_x, mask):
        N, L, _ = noise_x.shape
        mask = mask.repeat(noise_x.shape[0], 1)
        x = noise_x.masked_select(mask.bool().unsqueeze(-1))
        x = x.reshape(N, -1, self.mask_token.shape[-1])
        x = x + self.mask_token
        return x

    def forward_encoder(self, x, mask_ratio, kept_mask_ratio):
        x = self.grid_patchify(x)
        coords = None
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, kept_mask_ratio)

        # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :] # pos embed for cls token is 0 
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer blocks
        x_feats = []
        if self.use_input:
            x_feats.append(x)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if self.weight_fm and idx in self.use_fm:
                x_feats.append(x)

        if self.weight_fm:
            return x_feats, mask, ids_restore, coords
        else:
            x = self.norm(x)
            return x, mask, ids_restore, coords



    def forward_decoder(self, sequence, y, mask, ids_restore, coords, mask_ratio, kept_mask_ratio):
        """
        sequence: [N, 3, H, W]
        y: list of encoder features
        """
        
        N, L = ids_restore.shape
        L_mask = int(L * mask_ratio)
        
        # Ensure ids_restore is on the same device as y
        device = y[0].device

        # Create noise and visible mask
        noise = torch.randn(N, L_mask, 3, self.patch_size, self.patch_size, device=device)
        visible_mask = torch.zeros(N, L - L_mask, 3, self.patch_size, self.patch_size, device=device)
        noise = torch.cat((noise, visible_mask), dim=1)
        
        # Expand ids_restore to match noise dimensions
        ids_restore = ids_restore[:, :, None, None, None].expand(-1, -1, 3, self.patch_size, self.patch_size)
        if self.weight_fm:
            # y input: a list of Tensors (B, C, D)
            y = self.wfm(y)
        # Iterate over the batch
        results = []
        for n in range(sequence.shape[0]):
            b_ids_restore = ids_restore[n]
            b_mask = mask[n]
            _y = y[n].unsqueeze(0)
            # Perform the gather operation, ensuring dimensions match
            b_noise = torch.gather(noise[n], dim=0, index=b_ids_restore)
            
            # Reshape the noise into patches
            b_noise = rearrange(b_noise, '(h w) c s1 s2 -> c (h s1) (w s2)', 
                                s1=self.patch_size, s2=self.patch_size, 
                                h=self.img_size // self.patch_size, w=self.img_size // self.patch_size)
            
            b_batch = sequence[n].unsqueeze(0).expand(self.diffusion_timesteps, -1, -1, -1).to(y[0].device) # batch, channel, H, W
            # Add noise
            t = torch.arange(0, self.noise_scheduler.num_train_timesteps, dtype=torch.long, device=y[0].device).unsqueeze(1).expand(self.noise_scheduler.num_train_timesteps, -1)
            noise_x = self.noise_scheduler.add_noise(b_batch, b_noise, t)
            noise_x = self.decoder_grid_patchify(noise_x, t)
            x = self.mask_tokens_grid(noise_x, b_mask)
            
            for i, blk in enumerate(self.decoder_blocks):
                if self.weight_fm:
                    feat_y = _y[..., i].repeat(noise_x.shape[0], 1, 1)
                    x = blk(x, self.dec_norms[i](feat_y))
                else:
                    # NOTE: not using weighted feature maps
                    x = blk(x, _y)

            x = self.decoder_norm(x)
            x = self.decoder_pred(x) # N, L, patch_size**2 * 3

            results.append(x)

        # Stack results to form the final output
        x = torch.stack(results)
        return x, noise

    def forward_loss_noise(self, x, noise):
        """
            predict noise
        """
        loss = (x - noise) ** 2
        return loss
    
    def forward_loss(self, imgs, pred, mask, coords):
        """
        imgs: [N, 3, H, W]
        pred: [N, noise_batch, token_len, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        
        target = self.patchify(imgs)  # Convert images to patches
        target = target.masked_select(mask.bool().unsqueeze(-1)).reshape(target.shape[0], -1, target.shape[-1])  # Apply mask and reshape
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # Expand the second dimension of target to match the second dimension of pred
        target = target.unsqueeze(1).repeat(1, pred.shape[1], 1, 1)  # [N, noise_batch, token_len, p*p*3]

        # Compute loss
        loss = (pred - target) ** 2
        loss = loss.mean()

        return loss, target, pred

    def forward(self, imgs, mask_ratio=0.75, kept_mask_ratio=0.5):
        with torch.cuda.amp.autocast():
            latent, mask, ids_restore, coords = self.forward_encoder(imgs, mask_ratio, kept_mask_ratio)
            pred, noise = self.forward_decoder(imgs, latent, mask, ids_restore, coords, mask_ratio, kept_mask_ratio)  # [N, L, p*p*3]
            loss, target, pred = self.forward_loss(imgs, pred, mask, noise)
            # return loss, pred, mask
            return loss, target, pred


def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_small_patch16 = mae_vit_small_patch16_dec512d8b
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
