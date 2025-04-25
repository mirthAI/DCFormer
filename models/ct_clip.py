from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from models.ct_clip_utils import *
from models.utils.main_blocks import PreNorm, Attention, FeedForward, LayerNorm, PatchDropout, RotaryEmbedding
from models.convnext import LayerNorm
# checkpointing helper function
class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w z) c -> b c h w z', h = h_r, w= w_r)


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        causal = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        checkpoint_during_training = False
    ):
        super().__init__()
        self.checkpoint_during_training = checkpoint_during_training

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult)),
            ]))

        self.norm_in = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)

    def forward(
        self,
        x,
        rotary_pos_emb = None,
        mask = None
    ):
        can_checkpoint = self.training and self.checkpoint_during_training
        checkpoint_fn = make_checkpointable if can_checkpoint else identity

        x = self.norm_in(x)

        for attn, ff in self.layers:
            attn, ff = map(checkpoint_fn, (attn, ff))

            x = attn(x, mask, rotary_pos_emb) + x
            x = ff(x) + x

        return self.norm_out(x)

# text and vision transformers

class TextTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_tokens,
        max_seq_len,
        dim_head,
        rotary_pos_emb = None,
        causal = False,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if not rotary_pos_emb else None
        self.rotary_pos_emb = RotaryEmbedding(min(dim_head, 32)) if rotary_pos_emb else None

        self.cls_token = nn.Parameter(torch.randn(dim)) if not causal else None

        self.transformer = Transformer(dim, dim_head = dim_head, causal = causal, **kwargs)

    def forward(self, x, mask = None):
        b, n, device = *x.shape, x.device

        x = self.token_emb(x)

        if exists(self.abs_pos_emb):
            pos_emb = self.abs_pos_emb(torch.arange(n, device = device))
            x = x + rearrange(pos_emb, 'n d -> 1 n d')

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            rotary_pos_emb = self.rotary_pos_emb(n + 1, device = device)

        if exists(self.cls_token):
            cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = b)
            x = torch.cat((cls_tokens, x), dim = 1)

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

        out = self.transformer(x, mask = mask, rotary_pos_emb = rotary_pos_emb)
        return out

class VisionTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        image_size,
        patch_size,
        channels,
        patch_dropout = 0.5,
        **kwargs
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim)
        )

        self.pos_emb = nn.Embedding(num_patches, dim)
        self.patch_dropout = PatchDropout(patch_dropout)
        self.transformer = Transformer(dim, **kwargs)

        self.to_cls_tokens = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, dim, bias = False),
            Rearrange('b d -> b 1 d')
        )

    def forward(
        self,
        x,
        keep_all_patches = False
    ):
        device = x.device

        x = self.to_tokens(x)
        b, n, _ = x.shape

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> 1 n d')

        x = self.patch_dropout(x, force_keep_all = keep_all_patches)

        out = self.transformer(x)

        cls_tokens = self.to_cls_tokens(out)
        return torch.cat((cls_tokens, out), dim = 1)
# main clip class

# class GAP(nn.Module):
#     def __init__(self, dim_image, dim_latent) -> None:
#         super().__init__()
#         self.linear = nn.Linear(dim_image, dim_latent, bias = False)
#     def forward(self, x):
#         return self.linear(x.mean(dim=[-3, -2, -1]))
    
class LinearLatent(nn.Module):
    def __init__(self, dim_image, dim_latent) -> None:
        super().__init__()
        self.linear = nn.Linear(dim_image, dim_latent, bias = False)
    def forward(self, enc_image):
        enc_image = enc_image.view(enc_image.shape[0], -1)
        image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image
        image_embeds = self.linear(image_embeds)
        return image_embeds

class CTCLIP(nn.Module):
    def __init__(
        self,
        *,
        image_encoder = None,
        text_encoder = None,
        dim_text = 512,
        dim_image = 512,
        dim_latent = 512,
        num_text_tokens = 28897,
        text_enc_depth = 6,
        text_seq_len = 256,
        text_heads = 8,
        text_dim_head = 64,
        text_pad_id = 0,
        text_eos_id = None,
        visual_enc_depth = 6,
        visual_heads = 8,
        visual_dim_head = 64,
        visual_image_size = 256,
        visual_patch_size = 32,
        visual_patch_dropout = 0.5,
        reduction=None,
        visual_has_cls_token = False,
        channels = 3,
        downsample_image_embeds = False,
        use_mlm = False,
        text_ssl_loss_weight = 0.05,
        use_visual_ssl = False,
        image_ssl_loss_weight = 0.05,
        multiview_loss_weight = 0.1,
        checkpoint_during_training = False,
        **kwargs
    ):
        super().__init__()
        #assert use_all_token_embeds or (visual_has_cls_token or text_has_cls_token), 'CLS token must be included on both vision and text transformers if you are not using fine-grained contrastive learning loss'

        # store some parameters for access
        self.reduction = reduction
        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        self.image_channels = channels
        self.image_size = visual_image_size

        # instantiate text transformer

        self.text_pad_id = text_pad_id
        self.text_seq_len = text_seq_len


        self.text_eos_id = text_eos_id

        if exists(text_encoder):
            self.text_transformer = text_encoder
        else:
            self.text_transformer = TextTransformer(
                dim = dim_text,
                num_tokens = num_text_tokens + (1 if use_mlm else 0),
                max_seq_len = text_seq_len,
                depth = text_enc_depth,
                heads = text_heads,
                causal = False,
                dim_head = text_dim_head,
                rotary_pos_emb = False,
                checkpoint_during_training = checkpoint_during_training
            )

        # instantiate image transformer

        self.visual_has_cls_token = visual_has_cls_token

        if exists(image_encoder):
            self.visual_transformer = image_encoder
        else:
            self.visual_transformer = VisionTransformer(
                dim = dim_image,
                image_size = visual_image_size,
                patch_size = visual_patch_size,
                channels = channels,
                depth = visual_enc_depth,
                heads = visual_heads,
                dim_head = visual_dim_head,
                patch_dropout = visual_patch_dropout,
                checkpoint_during_training = checkpoint_during_training
            )

        # text ssl

        self.text_ssl_loss_weight = text_ssl_loss_weight if use_mlm else 0
        # image s
        self.image_ssl_loss_weight = image_ssl_loss_weight if use_visual_ssl else 0

        # text latent projection

        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)
        # self.to_text_latent = ProjectionHead(dim_text, dim_latent)

        # image latent projection

        # if downsample_image_embeds:
        #     #assert use_all_token_embeds, 'must be using all token embeds for contrastive learning in order to downsampling'
        #     dim_conv=512
        #     self.to_visual_latent = nn.Sequential(
        #         RearrangeImage(),
        #         nn.Conv3d(dim_conv, dim_conv, 4, stride = 2, padding = 1, bias = False, groups = dim_conv),
        #         nn.Conv3d(dim_conv, dim_latent, 1),
        #         Rearrange('b c h w z -> b (h w z c)'),
        #         nn.Linear(dim_image, dim_latent, bias = False)
        #         )
        # else:
      

        self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)  
        # self.to_visual_latent = ProjectionHead(dim_image, dim_latent)

        # if self.model_type.lower() == 'ctvit': self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)  
        #self.to_visual_latent = LinearLatent(dim_image, dim_latent) if model_type.lower() not in ['convnext', 'convnextf'] else GAP(dim_latent)



        # temperature

        # self.temperature = nn.Parameter(torch.tensor(1.))



    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        self.load_state_dict(pt)


    def forward(
        self,
        image,
        text,
        device,
        validation=False
        
    ):

        # derive text mask

        text_mask =text.attention_mask

        # concat augmented texts and images and do some asserts

        num_batch_texts = num_batch_images = 1
        #assert not (return_loss and not self.training), 'loss cannot be used if not training'
        # get encoded text

        text_args = (text.input_ids,text.attention_mask)

        text_args = (*text_args, text_mask)

        # print(text.input_ids.shape)

        text_embeddings = self.text_transformer(text.input_ids, attention_mask = text.attention_mask )
        enc_text = text_embeddings[0]

        # enc_image= self.visual_transformer(video=image, device=device)

        #print("This is visual encoding")
        # print(enc_image.shape)

        # global h_r, w_r, z_r
        # h_r, w_r, z_r = enc_image.shape[1], enc_image.shape[2], enc_image.shape[3]

        # if self.reduction == 'depth':
        #     enc_image = enc_image.mean(dim=1)

        # elif self.reduction == 'channel':
        #     enc_image = enc_image.mean(dim=[-4, -3, -2])

        # enc_image = enc_image.view(enc_image.shape[0], -1)    
            
        # #comment
        # enc_image = enc_image.mean(dim=[-4, -3, -2])

        # enc_image = enc_image[:, :] if enc_image.ndim == 3 else enc_image     
        # #comment

        # image_latents = self.to_visual_latent(enc_image)

        # project to latents
        #text_embeds = text_embeds.view(text_embeds.shape[0], -1)

                
        text_embeds = enc_text[:, :] if enc_text.ndim == 3 else enc_text
        text_embeds = text_embeds[:,0,:]
        text_latents = self.to_text_latent(text_embeds)

        text_latents = l2norm(text_latents)


        return None, text_latents
    # def forward( 
    #     self,
    #     image,
    #     text,
    #     device,
    #     validation=False
        
    # ):

    #     # derive text mask

    #     text_mask =text.attention_mask

    #     # concat augmented texts and images and do some asserts

    #     num_batch_texts = num_batch_images = 1
    #     #assert not (return_loss and not self.training), 'loss cannot be used if not training'
    #     # get encoded text

    #     text_args = (text.input_ids,text.attention_mask)

    #     text_args = (*text_args, text_mask)

    #     # print(text.input_ids.shape)

    #     text_embeddings = self.text_transformer(text.input_ids, attention_mask = text.attention_mask )
    #     enc_text = text_embeddings[0]

    #     enc_image= self.visual_transformer(video=image, device=device)

    #     #print("This is visual encoding")
    #     # print(enc_image.shape)

    #     global h_r, w_r, z_r
    #     h_r, w_r, z_r = enc_image.shape[1], enc_image.shape[2], enc_image.shape[3]

    #     if self.reduction == 'depth':
    #         enc_image = enc_image.mean(dim=1)

    #     elif self.reduction == 'channel':
    #         enc_image = enc_image.mean(dim=[-4, -3, -2])

    #     enc_image = enc_image.view(enc_image.shape[0], -1)    
            
    #     # #comment
    #     # enc_image = enc_image.mean(dim=[-4, -3, -2])

    #     enc_image = enc_image[:, :] if enc_image.ndim == 3 else enc_image     
    #     # #comment

    #     image_latents = self.to_visual_latent(enc_image)

    #     # project to latents
    #     #text_embeds = text_embeds.view(text_embeds.shape[0], -1)

                
    #     text_embeds = enc_text[:, :] if enc_text.ndim == 3 else enc_text
    #     text_embeds = text_embeds[:,0,:]
    #     text_latents = self.to_text_latent(text_embeds)

    #     text_latents, image_latents = map(l2norm, (text_latents, image_latents))
    #     temperature = nn.Parameter(torch.tensor(1.))
    #     temp = temperature.exp()

    #     if validation:
    #         return einsum('b d, b d -> b', text_latents, image_latents) * temp
    #     return image_latents, text_latents