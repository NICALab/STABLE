import argparse
import os
import itertools
import json
import random
from tqdm import tqdm
import warnings
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from datasets import *
from utils import *
from models import *

from skimage import io
import glob
from types import SimpleNamespace


# ignore user warnings
warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int, default=235000, help="epoch to test")
parser.add_argument("--exp_dir", type=str, default="./exp", help="path to experiments directory")
parser.add_argument("--gpu_id", type=int, default=0, help="gpu id to train on")

parser.add_argument("--img_dir", type=str, default="./path/to/images/to/test", help="path to test images")

opt_test = parser.parse_args()

if not os.path.exists(opt_test.exp_dir):
    print(f"Experiment directory: {opt_test.exp_dir} does not exist")
    exit()

exp_name = os.path.basename(opt_test.exp_dir)
print(f"Experiment name: {exp_name}")

Enc1_path = os.path.join(opt_test.exp_dir, "saved_models_" + exp_name, 'Enc1_'+str(opt_test.epoch)+'.pth')
Dec1_path = os.path.join(opt_test.exp_dir, "saved_models_" + exp_name, 'Dec1_'+str(opt_test.epoch)+'.pth')
Enc2_path = os.path.join(opt_test.exp_dir, "saved_models_" + exp_name, 'Enc2_'+str(opt_test.epoch)+'.pth')
Dec2_path = os.path.join(opt_test.exp_dir, "saved_models_" + exp_name, 'Dec2_'+str(opt_test.epoch)+'.pth')

saved_images_dir = os.path.join(opt_test.exp_dir, "images_" + exp_name, str(opt_test.epoch))
os.makedirs(saved_images_dir, exist_ok=True)
saved_images_in_dir = os.path.join(saved_images_dir, "in")
os.makedirs(saved_images_in_dir, exist_ok=True)
saved_images_trans_dir = os.path.join(saved_images_dir, "trans")
os.makedirs(saved_images_trans_dir, exist_ok=True)

opt_train_path = os.path.join(opt_test.exp_dir, "options_" + exp_name + ".json")
with open(opt_train_path, 'r') as json_file:
    opt = json.load(json_file)
opt = SimpleNamespace(**opt)

cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{opt_test.gpu_id}" if cuda else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

Enc1 = UNet(n_in=opt.n_ch_in, n_out=opt.n_ch_com, mid_channels=opt.G_mid_ch, norm_type=opt.G_norm_type, demodulated=opt.G_demodulated, act=opt.enc_act, momentum=opt.momentum)
Dec2 = UNet(n_in=opt.n_ch_com, n_out=opt.n_ch_out, mid_channels=opt.G_mid_ch, norm_type=opt.G_norm_type, demodulated=opt.G_demodulated, act=opt.dec_act, momentum=opt.momentum)

if cuda:
    print(f"Loading to device: {device}...")
    Enc1.to(device)
    Dec2.to(device)

Enc1.load_state_dict(torch.load(Enc1_path))
Dec2.load_state_dict(torch.load(Dec2_path))
Enc1.eval()
Dec2.eval()

opt.scale_ratio = (opt.scale_ratio[0], opt.scale_ratio[1])

data_test = glob.glob(f"{opt_test.img_dir}/*.*")
print(f"Number of test images: {len(data_test)}")

with torch.no_grad():
    for i in tqdm(range(len(data_test)), desc=f"Testing..."):
        img_path = data_test[i]
        X_1 = torch.from_numpy(io.imread(img_path).astype(np.float32)).to(device)
        if len(X_1.shape) == 2:
            X_1 = X_1.unsqueeze(0).unsqueeze(0)
        elif len(X_1.shape) == 3:
            X_1 = X_1.unsqueeze(0)

        Z_1 = Enc1(X_1)
        X_12 = Dec2(Z_1)

        filename = os.path.splitext(os.path.basename(img_path))[0]

        # Save images
        in_img_dir = os.path.join(saved_images_in_dir, f"{filename}_in_{i}.tif")
        trans_img_dir = os.path.join(saved_images_trans_dir, f"{filename}_trans_{i}.tif")
        io.imsave(in_img_dir, X_1.detach().cpu().numpy(), metadata={'axes': 'TYX'})
        io.imsave(trans_img_dir, X_12.detach().cpu().numpy(), metadata={'axes': 'TYX'})
