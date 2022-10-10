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

device='cuda'

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


def project( 
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device):
  
  assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

  def logprint(*args):
      if verbose:
          print(*args)

  G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore
  c = None

  # Compute w stats.
  logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
  z_samples = np.random.RandomState(123).randn(1, G.z_dim)

  print(z_samples.shape)

  wi = G.mapping(torch.tensor(z_samples, device=device), c=None)
  print(wi.shape)

  # diff = (wi[5] - wi[3]).sum().square()
  # print(diff)

  exit(0)


  # Load VGG16 feature detector.
  url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
  with dnnlib.util.open_url(url) as f:
      vgg16 = torch.jit.load(f).eval().to(device)

  # Features for target image.
  target_images = target.unsqueeze(0).to(device).to(torch.float32)
  if target_images.shape[2] > 256:
      target_images = F.interpolate(target_images, size=(256, 256), mode='area')
  target_features = vgg16(target_images, resize_images=False, return_lpips=True)

  z_opt = torch.tensor(z_samples, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
  z_out = torch.zeros([1] + list(z_opt.shape[:]), dtype=torch.float32, device=device)
  optimizer = torch.optim.Adam([z_opt], betas=(0.9, 0.999), lr=initial_learning_rate)
  
  noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }
  #for buf in noise_bufs.values():
  #    buf[:] = torch.randn_like(buf)
     # buf.requires_grad_(True) 
  
  for step in range(num_steps):
      # Learning rate schedule.
      t = step / num_steps
    #   w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
      lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
      lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
      lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
      lr = initial_learning_rate * lr_ramp
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

      ws = G.mapping(z_opt, c=None)
      ws.requires_grad_(True)
    
      curr_idx = 1
      curr_ws = ws[curr_idx].unsqueeze(0)
      left_ws = ws[curr_idx - 1].unsqueeze(0)
      right_ws = ws[curr_idx + 1].unsqueeze(0)
      curr_ws_context = torch.stack([left_ws, right_ws], dim=1)
      left_borders_idx=torch.tensor([0], requires_grad=False).to(device)

      synth_images = G.synthesis(curr_ws, ws_context=curr_ws_context, left_borders_idx=left_borders_idx, noise_mode='const')
      synth_images.requires_grad_(True)

      synth_images = (synth_images + 1) * (255/2)
      if synth_images.shape[2] > 256:
          synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

      # Features for synth images.
      synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
      dist = (target_features - synth_features).square().sum()

      loss = dist 
      loss.requires_grad_(True)

      # Step
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

      # Save projected W for each optimization step.
      z_out[0] = z_opt.detach()

  return z_out, noise_bufs


def run_project(network_pkl: str, target_fname: str, outdir: str, save_video: bool = True, seed: int = 303, num_steps: int = 1000):

  np.random.seed(seed)
  torch.manual_seed(seed)

  # Load networks.
  print('Loading networks from "%s"...' % network_pkl)
  device = torch.device('cuda')
  with dnnlib.util.open_url(network_pkl) as fp:
      G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
  G.mapping.num_modes = 0

  # Load target image.
  target_pil = PIL.Image.open(target_fname).convert('RGB')
  w, h = target_pil.size
  s = min(w, h)
  target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
  target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
  target_uint8 = np.array(target_pil, dtype=np.uint8)

  # Optimize projection.
  start_time = perf_counter()
  projected_z_steps, noise = project(
      G,
      target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
      num_steps=num_steps,
      device=device,
      verbose=True
  )

  print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

  # Render debug output: optional video and projected image and W vector.
#   os.makedirs(outdir, exist_ok=True)
#   if save_video:
#       video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
#       print (f'Saving optimization progress video "{outdir}/proj.mp4"')
#       for projected_z in projected_z_steps:


#           synth_image = get_image_from_w(G, projected_w)
#           synth_image = (synth_image + 1) * (255/2)
#           synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
#           video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
#       video.close()

  # Save final projected frame and W vector.
  target_pil.save(f'{outdir}/target.png')
  projected_z = projected_z_steps[-1]
  projected_w = G.mapping(projected_z, c=None)
  synth_image = get_image_from_w(G, projected_w)
  synth_image = (synth_image + 1) * (255/2)
  synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
  PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
 # np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())
  return projected_z, noise

#%%


network_pkl = 'https://kaust-cair.s3.amazonaws.com/alis/lhq1024-snapshot.pkl'

z, noise = run_project(network_pkl=network_pkl, 
           target_fname='./0003856.jpg', 
           outdir='./tmp/', num_steps=2000)

with open('./tmp/out.pkl', 'wb') as f:
	pkl.dump((z, noise), f)





