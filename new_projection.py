#%%
import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from PIL import Image

import PIL
import os
from PIL import Image
import argparse
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TVF
import numpy as np
from distutils.dir_util import copy_tree
from training.networks import SynthesisLayer
from training.networks import PatchWiseSynthesisLayer
import dnnlib
from scripts.legacy import load_network_pkl
import matplotlib.pyplot as plt

import dnnlib
import legacy
import pickle as pkl

from copy import deepcopy
#%%

network_pkl = 'https://kaust-cair.s3.amazonaws.com/alis/lhq1024-snapshot.pkl'
device='cuda'
print('Loading networks from "%s"...' % network_pkl)
device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as fp:
    G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
# G.mapping.num_modes = 0

#%%
def get_image_from_w(G, ws):
    curr_idx = 1
    curr_ws = ws[curr_idx].unsqueeze(0)
    left_ws = ws[curr_idx - 1].unsqueeze(0)
    right_ws = ws[curr_idx + 1].unsqueeze(0)
    curr_ws_context = torch.stack([left_ws, right_ws], dim=1)
    left_borders_idx=torch.tensor([0], requires_grad=False).to(device)

    synth_images = G.synthesis(curr_ws, ws_context=curr_ws_context, left_borders_idx=left_borders_idx, noise_mode='const')
    return synth_images

#%%
z = torch.randn((1, 512), device=device)
img = G(z, c=None)

print(img.shape)











# %%
