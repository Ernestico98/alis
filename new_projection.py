#%%
import copy
from logging import shutdown
import os
import shutil
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
from responses import target
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
    right_ws = torch.zeros_like(curr_ws).to(device)
    curr_ws_context = torch.stack([left_ws, right_ws], dim=1)
    left_borders_idx=torch.tensor([0], requires_grad=False).to(device)

    synth_images = G.synthesis(curr_ws, ws_context=curr_ws_context, left_borders_idx=left_borders_idx, noise_mode='const')
    return synth_images


def save_img(synth_image, path_to_save):
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(path_to_save)


def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device       = 'cuda'

):

    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore
    modes_idx = torch.from_numpy(np.array([0])).to(device)
    c = None
    
    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), c, skip_w_avg_update=True, modes_idx=modes_idx)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # print(w_samples.shape)
    # print(w_avg.shape)
    # print(w_std.shape)
    
    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device) # pylint: disable=not-callable
    w_opt = torch.cat([w_opt.clone(), w_opt.clone()], dim=0).requires_grad_(True)
    w_out = []
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = get_image_from_w(G, ws)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256: 
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')
        
        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        
        loss = dist + reg_loss * regularize_noise_weight
        loss.requires_grad_(True)
        
        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out.append(w_opt.detach().clone())

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out


#%%

def run_project(
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    out_name: str
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    w_out = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    end_time = perf_counter()
    print (f'Elapsed: {(end_time - start_time):.1f} s')

    #Save projection file
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, out_name + '.pkl'), 'wb') as f:
        pkl.dump(w_out, f)

    #Save original image
    shutil.copy(target_fname, os.path.join(outdir, out_name + '_original.jpg'))

    #Save video with projection steps
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in w_out:
            synth_image = get_image_from_w(G, projected_w.repeat([1, G.num_ws, 1]))
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    #Save last projected image
    ws = w_out[-1]
    ws = ws.repeat(1, G.num_ws, 1)
    print(ws.shape)
    img = get_image_from_w(G, ws)
    save_img(img, os.path.join(outdir, out_name + '.jpg'))


# run_project(
#     target_fname = '/home/ernestico/Desktop/Docet/Instagramers/jannikobenhoff/jannikobenhoff/posts/1/image1/original.jpg',
#     outdir = './ernesto_data/projection',
#     save_video = True,
#     seed = 123,
#     num_steps = 2000,
#     out_name = '1'
# )

# exit(0)

def run_project_on_folder(folder: str):
    for idx, file in enumerate(os.listdir(folder)):
        if file.split('.')[-1] != 'jpg':
            continue
        file_path = os.path.join(folder, file)
        run_project(
            target_fname = file_path,
            outdir = './ernesto_data/projection',
            save_video = False,
            seed = 123,
            num_steps = 3,
            out_name = f'{idx}'
        )


run_project_on_folder(folder = './ernesto_data/to_project')


exit(0)



#%%
torch.manual_seed(987)
c = None
# modes_idx = torch.randint(low=0, high=G.synthesis_cfg.num_modes, size=(1,), device=device)
modes_idx = torch.from_numpy(np.array([0])).to(device)

#%%
z = torch.randn((1, G.z_dim), device=device)
zl = torch.randn_like(z).to(device)
# zr = torch.randn_like(z).to(device)
zr = torch.zeros_like(z).to(device)

ws = G.mapping(zl, c, skip_w_avg_update=True, modes_idx=modes_idx)
wl = G.mapping(zl, c, skip_w_avg_update=True, modes_idx=modes_idx)
wr = torch.zeros_like(wl).to(device)

ws_context = torch.stack([wl, wr], dim=1)
# w_dist = int(0.5 * G.synthesis_cfg.patchwise.w_coord_dist * G.synthesis_cfg.patchwise.grid_size)
# left_borders_idx = torch.randint(low=0, high=(2 * w_dist - G.synthesis_cfg.patchwise.grid_size), size=z.shape[:1], device=device)
left_borders_idx = torch.tensor([0.0], device=device)
#when setting left_borders_idx to 0, then the right image is useless

#%%
img = G.synthesis(ws, ws_context=ws_context, left_borders_idx=left_borders_idx)
# img = G(z, c=None)
save_img(img, '/home/ernestico/Desktop/tmp.jpg')











# %%
