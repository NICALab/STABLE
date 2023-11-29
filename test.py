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



parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int, default=1800, help="epoch to test")
parser.add_argument("--exp_dir", type=str, default="/media/HDD4/josh/c2n/results/experiments/clean_test_id_5", help="path to experiments directory")
parser.add_argument("--gpu_id", type=int, default=0, help="gpu id to train on")
parser.add_argument("--seed", type=int, default=101, help="seed for random number generator")
parser.add_argument("--load_to_memory", type=int, default=0, help="load all data to memory")
parser.add_argument("--A_test_id", type=int, default=0, help="test id for domain A")
parser.add_argument("--B_test_id", type=int, default=0, help="test id for domain B")
parser.add_argument("--percentiles", type=list, default=[45.67070738302002, 76.07678560652094], help="percentiles for normalization")
parser.add_argument("--n_samples", type=int, default=-1, help="number of samples to test")

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
saved_images_tif_dir = f"{saved_images_dir}/tif/"
os.makedirs(saved_images_tif_dir, exist_ok=True)

opt_train_path = os.path.join(opt_test.exp_dir, "options_" + exp_name + ".json")
with open(opt_train_path, 'r') as json_file:
    opt = json.load(json_file)
opt = SimpleNamespace(**opt)

cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{opt_test.gpu_id}" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

Enc1 = Unet(input_channels=opt.input_channels, output_channels=opt.common_channels, upsampling_method="InterpolationConv", std=opt.std, norm=opt.model_norm, conv_type=opt.conv_type)
Dec1 = Unet(input_channels=opt.common_channels, output_channels=opt.input_channels, upsampling_method="InterpolationConv", norm=opt.model_norm, conv_type=opt.conv_type)
Enc2 = Unet(input_channels=opt.output_channels, output_channels=opt.common_channels, upsampling_method="InterpolationConv", std=opt.std, norm=opt.model_norm, conv_type=opt.conv_type)
Dec2 = Unet(input_channels=opt.common_channels, output_channels=opt.output_channels, upsampling_method="InterpolationConv", norm=opt.model_norm, conv_type=opt.conv_type)

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

A_dir = os.path.join(opt.dataset_dir, "test", "A")
B_dir = os.path.join(opt.dataset_dir, "test", "B")

# Get all files in A_dir and B_dir
A_files = glob.glob(os.path.join(A_dir, "*"))
B_files = glob.glob(os.path.join(B_dir, "*"))

X_1_full = io.imread(A_files[opt_test.A_test_id]).astype(np.float32)
X_2_full = io.imread(B_files[opt_test.B_test_id]).astype(np.float32)

# SKip first 10 frames and last 10 frames
X_1_full = X_1_full[10:-10, :, :]
X_2_full = X_2_full[10:-10, :, :]

for i in tqdm(range(min(X_1_full.shape[0], X_2_full.shape[0]))):

    X_1 = X_1_full[i, :, :]
    X_2 = X_2_full[i, :, :]

    X_1 = X_1[:256, :256]
    X_2 = X_2[:256, :256]
    # X_2 = X_2[480-256:480, 480-256:480]

    X_1 = np.clip(X_1, 0, None)
    X_2 = np.clip(X_2, 0, None)

    A_norm = opt_test.percentiles[0]
    B_norm = opt_test.percentiles[1]

    X_1 = X_1 / A_norm
    X_2 = X_2 / B_norm
    
    X_1 = torch.from_numpy(X_1).to(device)
    X_2 = torch.from_numpy(X_2).to(device)

    X_1 = X_1.unsqueeze(0)
    X_2 = X_2.unsqueeze(0)
    X_1 = X_1.unsqueeze(0)
    X_2 = X_2.unsqueeze(0)
    
    # -------------------------------
    #  Forward Passes
    # -------------------------------
    Z_1 = Enc1(X_1)
    Z_2 = Enc2(X_2)

    # Translate images
    X_12 = Dec2(Z_1)
    X_21 = Dec1(Z_2)
    
    X_11 = Dec1(Z_1)
    X_22 = Dec2(Z_2)

    # Cycle translation
    Z_12 = Enc2(X_12)
    Z_21 = Enc1(X_21)
    
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