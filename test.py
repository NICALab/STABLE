"""

TODO List:
1. Load data
2. Load models
3. Run test with test set

"""

import argparse
import os
import numpy as np
import math
import glob
from tqdm import tqdm
import itertools
import datetime
import time
import sys
import json
from tifffile import imwrite as imsavetiff
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from models import *
from datasets import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from datasets import ImageDataset
from skimage import io
# from pytorch_msssim import ssim, ms_ssim
import warnings
import re
import json
from types import SimpleNamespace
from unet import UNet


parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int, default=356, help="epoch to test")
parser.add_argument("--exp_dir", type=str, default="/media/HDD3/Cytosolic2NLS/results/041124_c2n_leica_nobg_256/experiments/041124_c2n_leica_nobg_256_multi_64_128_256_512_zch_3_10_2_10_100_0_0_0_0_0_0_0_0_0_1.0_1.2_minmax_trial2", help="path to experiments directory")
parser.add_argument("--gpu_id", type=int, default=0, help="gpu id to train on")
parser.add_argument("--A_test_id", type=int, default=0, help="test id for domain A")
parser.add_argument("--B_test_id", type=int, default=0, help="test id for domain B")
parser.add_argument("--n_samples", type=int, default=-1, help="number of samples to test")
parser.add_argument("--data_source", type=str, default="test", help="data source type")

opt_test = parser.parse_args()

if not os.path.exists(opt_test.exp_dir):
    print(f"Experiment directory: {opt_test.exp_dir} does not exist")
    exit()

exp_name = os.path.basename(opt_test.exp_dir)

Enc1_path = os.path.join(opt_test.exp_dir, "saved_models_" + exp_name, 'Enc1_'+str(opt_test.epoch)+'.pth')
Dec1_path = os.path.join(opt_test.exp_dir, "saved_models_" + exp_name, 'Dec1_'+str(opt_test.epoch)+'.pth')
Enc2_path = os.path.join(opt_test.exp_dir, "saved_models_" + exp_name, 'Enc2_'+str(opt_test.epoch)+'.pth')
Dec2_path = os.path.join(opt_test.exp_dir, "saved_models_" + exp_name, 'Dec2_'+str(opt_test.epoch)+'.pth')

saved_images_dir = os.path.join(opt_test.exp_dir, "images_" + exp_name, str(opt_test.epoch))
os.makedirs(saved_images_dir, exist_ok=True)
saved_images_tif_dir = f"{saved_images_dir}/{opt_test.n_samples}/"
os.makedirs(saved_images_tif_dir, exist_ok=True)

opt_train_path = os.path.join(opt_test.exp_dir, "options_" + exp_name + ".json")
with open(opt_train_path, 'r') as json_file:
    opt = json.load(json_file)
opt = SimpleNamespace(**opt)

cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{opt_test.gpu_id}" if cuda else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

if opt.lambda_kl_1 or opt.lambda_kl_2 > 0:
    kl_reg_1 = True
    kl_reg_2 = True
else:
    kl_reg_1 = False
    kl_reg_2 = False

if opt.conv_type == "DemodulatedConv2d":
    demodulated = True
elif opt.conv_type == "Conv2d":
    demodulated = False

Enc1 = UNet(n_channels=opt.input_channels, n_classes=opt.common_channels, mid_channels=opt.unet_mid_channels, bilinear=True, kl_reg=kl_reg_1, norm_type=opt.unet_norm_type, demodulated=demodulated)
Dec1 = UNet(n_channels=opt.common_channels, n_classes=opt.input_channels, mid_channels=opt.unet_mid_channels, bilinear=True, kl_reg=False, norm_type=opt.unet_norm_type, demodulated=demodulated)
Enc2 = UNet(n_channels=opt.output_channels, n_classes=opt.common_channels, mid_channels=opt.unet_mid_channels, bilinear=True, kl_reg=kl_reg_1, norm_type=opt.unet_norm_type, demodulated=demodulated)
Dec2 = UNet(n_channels=opt.common_channels, n_classes=opt.output_channels, mid_channels=opt.unet_mid_channels, bilinear=True, kl_reg=False, norm_type=opt.unet_norm_type, demodulated=demodulated)

if cuda:
    print(f"Loading to device: {device}...")
    Enc1.to(device)
    Dec1.to(device)
    Enc2.to(device)
    Dec2.to(device)

Enc1.load_state_dict(torch.load(Enc1_path))
Dec1.load_state_dict(torch.load(Dec1_path))
Enc2.load_state_dict(torch.load(Enc2_path))
Dec2.load_state_dict(torch.load(Dec2_path))
Enc1.eval()
Dec1.eval()
Enc2.eval()
Dec2.eval()

A_dir = os.path.join(opt.dataset_dir, opt_test.data_source, "A")
B_dir = os.path.join(opt.dataset_dir, opt_test.data_source, "B")

A_dir = "/media/HDD1/Cytosolic2NLS/data/040524_c2n_leica_256/test/A"

# Get all files in A_dir and B_dir
A_files = glob.glob(os.path.join(A_dir, "*"))
B_files = glob.glob(os.path.join(B_dir, "*"))

scale_ratio=(1.0, 1.2)
eps = 1e-7

X_1_full = torch.from_numpy(io.imread(A_files[opt_test.A_test_id])).float()
X_1_full = torch.nn.functional.interpolate(X_1_full.unsqueeze(0), scale_factor=(scale_ratio[0], scale_ratio[0]), mode='bilinear').squeeze()
X_1_full = np.clip(X_1_full, 0, None)
X_1_full = torch.clamp(X_1_full, 0, np.percentile(X_1_full, 99))
X_1_full = X_1_full / (X_1_full.max() + eps)
X_1_full = X_1_full[:, :opt.crop_size, :opt.crop_size]

X_2_full = torch.from_numpy(io.imread(B_files[opt_test.B_test_id])).float()
X_2_full = torch.nn.functional.interpolate(X_2_full.unsqueeze(0), scale_factor=(scale_ratio[1], scale_ratio[1]), mode='bilinear').squeeze()
X_2_full = np.clip(X_2_full, 0, None)
X_2_full = torch.clamp(X_2_full, 0, np.percentile(X_2_full, 99))
X_2_full = X_2_full / (X_2_full.max() + eps)
X_2_full = X_2_full[:, :opt.crop_size, :opt.crop_size]
# SKip first 10 frames and last 10 frames
# X_1_full = X_1_full[10:-10, :, :]
# X_2_full = X_2_full[10:-10, :, :]

X_1_save = np.zeros_like(X_1_full.numpy())
X_2_save = np.zeros_like(X_2_full.numpy())
X_12_save = np.zeros_like(X_1_full.numpy())
X_21_save = np.zeros_like(X_2_full.numpy())
X_121_save = np.zeros_like(X_1_full.numpy())
X_212_save = np.zeros_like(X_2_full.numpy())

for i in tqdm(range(min(X_1_full.shape[0], X_2_full.shape[0]))):

    X_1 = X_1_full[i, :, :].to(device)
    X_2 = X_2_full[i, :, :].to(device)

    X_1 = X_1.unsqueeze(0)
    X_2 = X_2.unsqueeze(0)
    X_1 = X_1.unsqueeze(0)
    X_2 = X_2.unsqueeze(0)
    
    # Iniital feature maps
    if opt.lambda_kl_1 > 0 or opt.lambda_kl_2 > 0:
        mu_Z_1, Z_1 = Enc1(X_1)
        mu_Z_2, Z_2 = Enc2(X_2)
    else:
        Z_1 = Enc1(X_1)
        Z_2 = Enc2(X_2)

    # Reconstruction
    X_11 = Dec1(Z_1)
    X_22 = Dec2(Z_2)

    # Translate images to opposite domain
    X_12 = Dec2(Z_1)
    X_21 = Dec1(Z_2)

    # Translated feature maps
    if opt.lambda_kl_1 > 0 or opt.lambda_kl_2 > 0:
        mu_Z_12, Z_12 = Enc2(X_12)
        mu_Z_21, Z_21 = Enc1(X_21)
    else:
        Z_12 = Enc2(X_12)
        Z_21 = Enc1(X_21)

    # Cycle reconstruction
    X_121 = Dec1(Z_12)
    X_212 = Dec2(Z_21)

    X_1 = X_1[0].cpu().float().detach().numpy()
    Z_1 = Z_1[0].cpu().float().detach().numpy()
    X_12 = X_12[0].cpu().float().detach().numpy()
    Z_12 = Z_12[0].cpu().float().detach().numpy()
    X_2 = X_2[0].cpu().float().detach().numpy()
    Z_2 = Z_2[0].cpu().float().detach().numpy()
    X_21 = X_21[0].cpu().float().detach().numpy()
    Z_21 = Z_21[0].cpu().float().detach().numpy()
    X_121 = X_121[0].cpu().float().detach().numpy()
    X_212 = X_212[0].cpu().float().detach().numpy()
    X_11 = X_11[0].cpu().float().detach().numpy()
    X_22 = X_22[0].cpu().float().detach().numpy()

    os.makedirs(f"{saved_images_tif_dir}/X_1/", exist_ok=True)
    os.makedirs(f"{saved_images_tif_dir}/Z_1/", exist_ok=True)
    os.makedirs(f"{saved_images_tif_dir}/X_12/", exist_ok=True)
    os.makedirs(f"{saved_images_tif_dir}/Z_12/", exist_ok=True)
    os.makedirs(f"{saved_images_tif_dir}/X_2/", exist_ok=True)
    os.makedirs(f"{saved_images_tif_dir}/Z_2/", exist_ok=True)
    os.makedirs(f"{saved_images_tif_dir}/X_21/", exist_ok=True)
    os.makedirs(f"{saved_images_tif_dir}/Z_21/", exist_ok=True)
    os.makedirs(f"{saved_images_tif_dir}/X_121/", exist_ok=True)
    os.makedirs(f"{saved_images_tif_dir}/X_212/", exist_ok=True)
    os.makedirs(f"{saved_images_tif_dir}/X_11/", exist_ok=True)
    os.makedirs(f"{saved_images_tif_dir}/X_22/", exist_ok=True)

    io.imsave(f"{saved_images_tif_dir}/X_1/X_1_{str(i).zfill(4)}.tif", X_1, metadata={'axes': 'TYX'})
    # io.imsave(f"{saved_images_tif_dir}/Z_1/Z_1_{str(i).zfill(4)}.tif", Z_1, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/X_12/X_12_{str(i).zfill(4)}.tif", X_12, metadata={'axes': 'TYX'})
    # io.imsave(f"{saved_images_tif_dir}/Z_12/Z_12_{str(i).zfill(4)}.tif", Z_12, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/X_2/X_2_{str(i).zfill(4)}.tif", X_2, metadata={'axes': 'TYX'})
    # io.imsave(f"{saved_images_tif_dir}/Z_2/Z_2_{str(i).zfill(4)}.tif", Z_2, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/X_21/X_21_{str(i).zfill(4)}.tif", X_21, metadata={'axes': 'TYX'})
    # io.imsave(f"{saved_images_tif_dir}/Z_21/Z_21_{str(i).zfill(4)}.tif", Z_21, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/X_121/X_121_{str(i).zfill(4)}.tif", X_121, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/X_212/X_212_{str(i).zfill(4)}.tif", X_212, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/X_11/X_11_{str(i).zfill(4)}.tif", X_11, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/X_22/X_22_{str(i).zfill(4)}.tif", X_22, metadata={'axes': 'TYX'})
    
    for ch in range(Z_1.shape[0]):
        io.imsave(f"{saved_images_tif_dir}/Z_1/Z_1_{str(i).zfill(4)}_ch_{ch}.tif", Z_1[ch, :, :], metadata={'axes': 'TYX'})
        io.imsave(f"{saved_images_tif_dir}/Z_12/Z_12_{str(i).zfill(4)}_ch_{ch}.tif", Z_12[ch, :, :], metadata={'axes': 'TYX'})
        io.imsave(f"{saved_images_tif_dir}/Z_2/Z_2_{str(i).zfill(4)}_ch_{ch}.tif", Z_2[ch, :, :], metadata={'axes': 'TYX'})
        io.imsave(f"{saved_images_tif_dir}/Z_21/Z_21_{str(i).zfill(4)}_ch_{ch}.tif", Z_21[ch, :, :], metadata={'axes': 'TYX'})
    
    if opt_test.n_samples != -1:
        if i == opt_test.n_samples:
            break
