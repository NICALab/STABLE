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

from torchvision.utils import save_image


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


if opt.data_type == "c2n":
    val_dataloader = DataLoader(
        ImageDataset(base_dataset_dir=opt.dataset_dir, mode="test", normalize=opt.normalize, seed=opt.seed, size=opt.crop_size, augmentation=False, scale_ratio=opt.scale_ratio),
        batch_size=1, num_workers=1, shuffle=False, drop_last=True
    )
elif opt.data_type == "stain":
    val_dataloader = DataLoader(
        HnEDataset(base_dataset_dir=opt.dataset_dir, mode="test", normalize=opt.normalize, seed=opt.seed, size=opt.crop_size, augmentation=False, scale_ratio=opt.scale_ratio),
        batch_size=1, num_workers=1, shuffle=False, drop_last=True
    )

print("Length of Val Dataloader: ", len(val_dataloader))

with torch.no_grad():
    # for i, batch in enumerate(tqdm(val_dataloader, desc=f"Running inference", position=1, leave=False)):
    for i, batch in enumerate(val_dataloader):
        X_1 = Variable(batch["A"].type(Tensor)).to(device)
        X_1_path = batch["path_A"][0]
        if len(X_1.shape) == 2:
            X_1 = X_1.unsqueeze(0).unsqueeze(0)
        elif len(X_1.shape) == 3:
            X_1 = X_1.unsqueeze(0)

        Z_1 = Enc1(X_1)
        X_12 = Dec2(Z_1)

        filename = os.path.splitext(os.path.basename(X_1_path))[0]

        # Save images
        in_img_dir = os.path.join(saved_images_in_dir, f"{filename}_in_{i}.png")
        trans_img_dir = os.path.join(saved_images_trans_dir, f"{filename}_trans_{i}.png")

        # save using save_image
        save_image(X_1, in_img_dir)
        save_image(X_12, trans_img_dir)
