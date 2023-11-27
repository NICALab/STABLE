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
from skimage import io
# from pytorch_msssim import ssim, ms_ssim
from logger import setup_logger
import warnings
import re
import json



parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int, default=9100, help="epoch to test")
# parser.add_argument("--dataset_name", type=str, default="Cytosolic2NLS_dataset_512_5x_10000_10000", help="name of the dataset")
parser.add_argument("--experiment_name", type=str, default="231025_n2c_only_crec_1_0.1_crec_2_10_irec_1_1_irec_2_100_iadv_1_10_iadv_2_1_icyc_1_0.1_icyc_2_10_seed_101_dweight_0", help="name of the experiment")

parser.add_argument("--test_type", type=str, default="full", help="Type of test (inference) [full | translation]")

parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")

parser.add_argument("--n_samples", type=int, default=-1, help="number downsampling layers in encoder")

parser.add_argument("--in_channels", type=int, default=1, help="number of channels of input images")
parser.add_argument("--out_channels", type=int, default=1, help="number of channels of output images")
parser.add_argument("--feat_channels", type=int, default=1, help="number of channels of feature")

parser.add_argument("--data_A_path", type=str, default='/home/yedam/workspace/Vanilla_Cytosolic2NLS/A_Test.tif', help="Path to test data for domain A")
parser.add_argument("--data_B_path", type=str, default='/home/yedam/workspace/Vanilla_Cytosolic2NLS/B_test.tif', help="Path to test data for domain B")

parser.add_argument("--gpu_id", type=int, default=1, help="gpu id to train on")

opt = parser.parse_args()

experiment_dir = os.path.join(f"/media/HDD1/josh_hdd/c2n/results/231022_n2c_dynamic_weights/experiments", opt.experiment_name)
if not os.path.exists(experiment_dir):
    print(experiment_dir)
    print("Experiment directory does not exist")
    exit()

Enc1_path = os.path.join(experiment_dir, "saved_models_" + opt.experiment_name, 'Enc1_'+str(opt.epoch)+'.pth')
Dec1_path = os.path.join(experiment_dir, "saved_models_" + opt.experiment_name, 'Dec1_'+str(opt.epoch)+'.pth')
Enc2_path = os.path.join(experiment_dir, "saved_models_" + opt.experiment_name, 'Enc2_'+str(opt.epoch)+'.pth')
Dec2_path = os.path.join(experiment_dir, "saved_models_" + opt.experiment_name, 'Dec2_'+str(opt.epoch)+'.pth')

saved_images_dir = os.path.join(experiment_dir, "images_" + opt.experiment_name, str(opt.epoch))
os.makedirs(saved_images_dir, exist_ok=True)
saved_images_tif_dir = f"{saved_images_dir}/tif/"
os.makedirs(saved_images_tif_dir, exist_ok=True)

model_opt_path = os.path.join(experiment_dir, "options_" + opt.experiment_name + ".json")
with open(model_opt_path, 'r') as json_file:
    model_opt = json.load(json_file)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Define model and load model checkpoint
# Initialize encoders, generators and discriminators
# Enc1 = Unet(input_channels=1, output_channels=1, upsampling_method="InterpolationConv", std=0, norm=0)
# Dec1 = Unet(input_channels=1, output_channels=1, upsampling_method="InterpolationConv", std=0, norm=0)
# Enc2 = Unet(input_channels=1, output_channels=1, upsampling_method="InterpolationConv", std=0, norm=0)
# Dec2 = Unet(input_channels=1, output_channels=1, upsampling_method="InterpolationConv", std=0, norm=0)

Enc1 = Unet(input_channels=1, output_channels=1, upsampling_method="InterpolationConv", std=model_opt["std"], norm=model_opt["model_norm"], conv_type=model_opt["conv_type"])
Dec1 = Unet(input_channels=1, output_channels=1, upsampling_method="InterpolationConv", norm=model_opt["model_norm"], conv_type=model_opt["conv_type"])
Enc2 = Unet(input_channels=1, output_channels=1, upsampling_method="InterpolationConv", std=model_opt["std"], norm=model_opt["model_norm"], conv_type=model_opt["conv_type"])
Dec2 = Unet(input_channels=1, output_channels=1, upsampling_method="InterpolationConv", norm=model_opt["model_norm"], conv_type=model_opt["conv_type"])

cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{opt.gpu_id}")
if cuda:
    print(device)
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

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

X_1_full = io.imread(opt.data_A_path).astype(np.float32)
X_2_full = io.imread(opt.data_B_path).astype(np.float32)



for i in tqdm(range(X_1_full.shape[0])):

    X_1 = X_1_full[i, :, :]
    X_2 = X_2_full[i, :, :]

    # X_1 = X_1[:256, :256]
    # X_2 = X_2[480-256:480, 480-256:480]

    X_1 = np.clip(X_1, 0, None)
    X_2 = np.clip(X_2, 0, None)

    eps = 1e-6
    A_norm = np.nanpercentile(X_1, 95) + eps
    B_norm = np.nanpercentile(X_2, 95) + eps

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
    io.imsave(f"{saved_images_tif_dir}/Z_1/Z_1_{str(i).zfill(4)}.tif", Z_1, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/X_12/X_12_{str(i).zfill(4)}.tif", X_12, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/Z_12/Z_12_{str(i).zfill(4)}.tif", Z_12, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/X_2/X_2_{str(i).zfill(4)}.tif", X_2, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/Z_2/Z_2_{str(i).zfill(4)}.tif", Z_2, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/X_21/X_21_{str(i).zfill(4)}.tif", X_21, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/Z_21/Z_21_{str(i).zfill(4)}.tif", Z_21, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/X_121/X_121_{str(i).zfill(4)}.tif", X_121, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/X_212/X_212_{str(i).zfill(4)}.tif", X_212, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/X_11/X_11_{str(i).zfill(4)}.tif", X_11, metadata={'axes': 'TYX'})
    io.imsave(f"{saved_images_tif_dir}/X_22/X_22_{str(i).zfill(4)}.tif", X_22, metadata={'axes': 'TYX'})
    
    if opt.n_samples != -1:
        if i == opt.n_samples:
            break