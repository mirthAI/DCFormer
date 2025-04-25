__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange, repeat, reduce


# class ClipLoss(nn.Module):
#     def __init__(self, device='cuda') -> None:
#         super().__init__()
#         # self.scale = scale
#         self.device = device
#         self.loss_img = nn.CrossEntropyLoss()
#         self.loss_txt = nn.CrossEntropyLoss()

#     def forward(self, logits_img, logits_txt, gt):
#         loss = (self.loss_img(logits_img, gt) + self.loss_txt(logits_txt, gt)) / 2        
#         return loss
def log(t, eps = 1e-20):
    return torch.log(t + eps)

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

class ClipLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, image_latents, text_latents, device='cuda'):
        temperature = nn.Parameter(torch.tensor(1.))
        temp = temperature.exp()

        text_latents = rearrange(text_latents, '(m b) ... -> m b ...', m = 1)
        image_latents = rearrange(image_latents, '(m b) ... -> m b ...', m = 1)
        text_to_image = einsum('m t d, n i d -> m n t i', text_latents, image_latents) * temp
        image_to_text = rearrange(text_to_image, '... t i -> ... i t')

        text_to_image = rearrange(text_to_image, 'm n ... -> (m n) ...')
        image_to_text = rearrange(image_to_text, 'm n ... -> (m n) ...')


        # exponentiate
        text_to_image_exp, image_to_text_exp = map(torch.exp, (text_to_image, image_to_text))

        # print(text_to_image_exp)
        # numerators
        text_to_image_pos, image_to_text_pos = map(matrix_diag, (text_to_image_exp, image_to_text_exp))

        text_to_image_denom, image_to_text_denom = map(lambda t: t.sum(dim = -1), (text_to_image_exp, image_to_text_exp))
        # loss

        text_to_image_loss = (-log(text_to_image_pos) + log(text_to_image_denom)).mean(dim = -1)
        image_to_text_loss = (-log(image_to_text_pos) + log(image_to_text_denom)).mean(dim = -1)

        # calculate CL loss

        cl_losses = (text_to_image_loss + image_to_text_loss) / 2

        loss = cl_losses[0]
        
        return loss
