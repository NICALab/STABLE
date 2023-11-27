import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys
import json
from tqdm import tqdm
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
from sklearn.metrics import mutual_info_score
# from pytorch_msssim import ssim, ms_ssim
import warnings
import random
import PIL


warnings.filterwarnings("ignore", category=UserWarning)

PIL.Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser()

# Training parameters
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100000000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")

parser.add_argument("--lr_G", type=float, default=3e-4, help="adam: learning rate")
parser.add_argument("--lr_D_1", type=float, default=3e-4, help="adam: learning rate")
parser.add_argument("--lr_D_2", type=float, default=3e-4, help="adam: learning rate")

parser.add_argument("--weight_decay", type=float, default=0, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--step_size", type=int, default=1000, help="adam: decay of first order momentum of gradient")
parser.add_argument("--gamma", type=float, default=0.1, help="how much to decay learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--gpu_id", type=int, default=1, help="gpu id to train on")
parser.add_argument("--seed", type=int, default=101, help="seed, None if you want a random seed")

# Logging parameters
parser.add_argument("--exp_name", type=str, default="", help="name of the experiment")
parser.add_argument("--output_dir", type=str, default="./results", help="output directory")

parser.add_argument("--log_interval", type=int, default=500, help="interval saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")

# Dataset parameters
parser.add_argument("--dataset_dir", type=str, default="/media/HDD2/Josh/Cytosolic2NLS/Data/230427_Live_to_HnE")
parser.add_argument("--datatype", type=str, default="tif", help="[tif | png]")
parser.add_argument("--crop_size", type=int, default=256, help="Crop size")
parser.add_argument("--augmentation", type=int, default=1, help="augment data")
parser.add_argument("--normalize", type=str, default="dataset", help="[ dataset | data ] normalize data against entire dataset = dataset, or normalize each image = data")
parser.add_argument("--load_to_memory", type=int, default=0, help="load data to memory")

# Model parameters
parser.add_argument("--num_scales", type=int, default=3, help="number of scales for the multi-scale discriminator")
parser.add_argument("--downsample_stride", type=int, default=2, help="stride for downsample for the multi-scale discriminator (ex: 2=1/2, 4=1/4)")
parser.add_argument("--input_channels", type=int, default=3, help="number of channels in the input image")
parser.add_argument("--output_channels", type=int, default=3, help="number of channels in the output image")
parser.add_argument("--common_channels", type=int, default=3, help="number of channels in the common feature")
parser.add_argument("--std", type=int, default=0, help="std")
parser.add_argument("--conv_type", type=str, default="DemodulatedConv2d", help="[Conv2d | DemodulatedConv2d]")
parser.add_argument("--model_norm", type=int, default=0, help="normalize the model")

# Loss parameters
parser.add_argument("--lambda_img_adv_1", type=float, default=10, help="weight for image domain adversarial loss")
parser.add_argument("--lambda_img_adv_2", type=float, default=10, help="weight for image domain adversarial loss")
parser.add_argument("--lambda_com_rec_1", type=float, default=1, help="weight for common feature reconstruction loss")
parser.add_argument("--lambda_com_rec_2", type=float, default=1, help="weight for common feature reconstruction loss")
parser.add_argument("--lambda_img_rec_1", type=float, default=100, help="weight for image reconstruction loss")
parser.add_argument("--lambda_img_rec_2", type=float, default=100, help="weight for image reconstruction loss")
parser.add_argument("--lambda_img_cyc_1", type=float, default=0, help="weight for image cycle reconstruction loss")
parser.add_argument("--lambda_img_cyc_2", type=float, default=0, help="weight for image cycle reconstruction loss")

opt = parser.parse_args()
print(opt)
options_dict = vars(opt)

# Fix Seed
if opt.seed != None:
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

# Setup Directories
experiment_dir = os.path.join(opt.output_dir, "experiments", opt.exp_name)
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

saved_images_dir = os.path.join(experiment_dir, f"images_{opt.exp_name}")
os.makedirs(saved_images_dir, exist_ok=True)

saved_models_dir = os.path.join(experiment_dir, f"saved_models_{opt.exp_name}")
os.makedirs(saved_models_dir, exist_ok=True)

options_save_file = os.path.join(experiment_dir, f"options_{opt.exp_name}.json")
with open(options_save_file, 'w') as file:
    json.dump(options_dict, file)

writer_logdir = os.path.join(experiment_dir, f"runs_{opt.exp_name}")
os.makedirs(writer_logdir, exist_ok=True)

# Setup writer for tensorboard
writer = SummaryWriter(log_dir=writer_logdir)
writer.add_text('Options', str(opt), 0)

Enc1 = Unet(input_channels=opt.input_channels, output_channels=opt.common_channels, upsampling_method="InterpolationConv", std=opt.std, norm=opt.model_norm, conv_type=opt.conv_type)
Dec1 = Unet(input_channels=opt.common_channels, output_channels=opt.input_channels, upsampling_method="InterpolationConv", norm=opt.model_norm, conv_type=opt.conv_type)
Enc2 = Unet(input_channels=opt.output_channels, output_channels=opt.common_channels, upsampling_method="InterpolationConv", std=opt.std, norm=opt.model_norm, conv_type=opt.conv_type)
Dec2 = Unet(input_channels=opt.common_channels, output_channels=opt.output_channels, upsampling_method="InterpolationConv", norm=opt.model_norm, conv_type=opt.conv_type)

D1 = MultiDiscriminator(channels=opt.input_channels, num_scales=opt.num_scales, downsample_stride=opt.downsample_stride)
D2 = MultiDiscriminator(channels=opt.output_channels, num_scales=opt.num_scales, downsample_stride=opt.downsample_stride)

criterion_recon = torch.nn.L1Loss()

cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{opt.gpu_id}")
if cuda:
    Enc1.to(device)
    Dec1.to(device)
    Enc2.to(device)
    Dec2.to(device)
    D1.to(device)
    D2.to(device)
    criterion_recon.to(device)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters()),
    lr=opt.lr_G,
    betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay
)
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr_D_1, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr_D_2, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=opt.step_size,
                                        gamma=opt.gamma, last_epoch=-1)
lr_scheduler_D1 = torch.optim.lr_scheduler.StepLR(optimizer_D1, step_size=opt.step_size,
                                        gamma=opt.gamma, last_epoch=-1)
lr_scheduler_D2 = torch.optim.lr_scheduler.StepLR(optimizer_D2, step_size=opt.step_size,
                                        gamma=opt.gamma, last_epoch=-1)

if opt.epoch != 0:
    # Load pretrained models
    Enc1.load_state_dict(torch.load(f"{saved_models_dir}/Enc1_{opt.epoch}.pth"))
    Dec1.load_state_dict(torch.load(f"{saved_models_dir}/Dec1_{opt.epoch}.pth"))
    Enc2.load_state_dict(torch.load(f"{saved_models_dir}/Enc2_{opt.epoch}.pth"))
    Dec2.load_state_dict(torch.load(f"{saved_models_dir}/Dec2_{opt.epoch}.pth"))
    D1.load_state_dict(torch.load(f"{saved_models_dir}/D1_{opt.epoch}.pth"))
    D2.load_state_dict(torch.load(f"{saved_models_dir}/D2_{opt.epoch}.pth"))
    optimizer_G.load_state_dict(torch.load(f"{saved_models_dir}/optimizer_G_{opt.epoch}.pth"))
    optimizer_D1.load_state_dict(torch.load(f"{saved_models_dir}/optimizer_D1_{opt.epoch}.pth"))
    optimizer_D2.load_state_dict(torch.load(f"{saved_models_dir}/optimizer_D2_{opt.epoch}.pth"))
    lr_scheduler_G.load_state_dict(torch.load(f"{saved_models_dir}/lr_scheduler_G_{opt.epoch}.pth"))
    lr_scheduler_D1.load_state_dict(torch.load(f"{saved_models_dir}/lr_scheduler_D1_{opt.epoch}.pth"))
    lr_scheduler_D2.load_state_dict(torch.load(f"{saved_models_dir}/lr_scheduler_D2_{opt.epoch}.pth"))
else:
    # Initialize weights
    Enc1.apply(weights_init('kaiming'))
    Dec1.apply(weights_init('kaiming'))
    Enc2.apply(weights_init('kaiming'))
    Dec2.apply(weights_init('kaiming'))
    D1.apply(weights_init('gaussian'))
    D2.apply(weights_init('gaussian'))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Dataset loader
train_dataloader = DataLoader(
    ImageDataset(base_dataset_dir=opt.dataset_dir, mode="train", normalize=opt.normalize, datatype='tif', seed=opt.seed, load_to_memory=True, size=opt.crop_size, test_idx=None),
    batch_size=opt.batch_size, num_workers=opt.n_cpu, shuffle=True
)

val_dataloader = DataLoader(
    ImageDataset(base_dataset_dir=opt.dataset_dir, mode="test", normalize=opt.normalize, datatype='tif', seed=opt.seed, load_to_memory=True, size=opt.crop_size, test_idx=None),
    batch_size=1, num_workers=1, shuffle=True
)

def compute_loss_D_adv(model, x, gt):
    """Computes the MSE between model output and scalar gt"""
    loss = 0
    # n=0
    output = model.forward(x)
    for out in output:
        squared_diff = (out - gt) ** 2
        loss += torch.mean(squared_diff)
        # n=n+1
    return loss

# Adversarial ground truths
valid = 1
fake = 0

tot_i = 0
# Train loop
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch}")):
        Enc1.train()
        Dec1.train()
        Enc2.train()
        Dec2.train()

        # Input data
        X_1 = Variable(batch["A"].type(Tensor)).to(device)
        X_2 = Variable(batch["B"].type(Tensor)).to(device)

        # Iniital feature maps
        Z_1 = Enc1(X_1)
        Z_2 = Enc2(X_2)

        # Reconstruction
        X_11 = Dec1(Z_1)
        X_22 = Dec2(Z_2)

        # Translate images to opposite domain
        X_12 = Dec2(Z_1)
        X_21 = Dec1(Z_2)

        # Translated feature maps
        Z_12 = Enc2(X_12)
        Z_21 = Enc1(X_21)

        # Cycle reconstruction
        X_121 = Dec1(Z_12)
        X_212 = Dec2(Z_21)

        # Compute Losses
        optimizer_G.zero_grad()

        # Image Domain Adversarial Losses
        loss_adv_1 = opt.lambda_img_adv_1 * compute_loss_D_adv(D1, X_21, valid)
        loss_adv_2 = opt.lambda_img_adv_2 * compute_loss_D_adv(D2, X_12, valid)

        # Common feature map reconstruction loss
        loss_com_rec_1 = opt.lambda_com_rec_1 * criterion_recon(Z_12, Z_1.detach()) if opt.lambda_com_rec_1 > 0 else 0
        loss_com_rec_2 = opt.lambda_com_rec_2 * criterion_recon(Z_21, Z_2.detach()) if opt.lambda_com_rec_2 > 0 else 0

        # Image reconstruction loss
        loss_img_rec_1 = opt.lambda_img_rec_1 * criterion_recon(X_11, X_1) if opt.lambda_img_rec_1 > 0 else 0
        loss_img_rec_2 = opt.lambda_img_rec_2 * criterion_recon(X_22, X_2) if opt.lambda_img_rec_2 > 0 else 0

        # Image cycle reconstruction loss
        loss_img_cyc_1 = opt.lambda_img_cyc_1 * criterion_recon(X_121, X_1) if opt.lambda_img_cyc_1 > 0 else 0
        loss_img_cyc_2 = opt.lambda_img_cyc_2 * criterion_recon(X_212, X_2) if opt.lambda_img_cyc_2 > 0 else 0 

        # Total loss
        loss_G = (
            loss_adv_1
            + loss_adv_2
            + loss_com_rec_1
            + loss_com_rec_2
            + loss_img_rec_1
            + loss_img_rec_2
            + loss_img_cyc_1
            + loss_img_cyc_2
        )

        loss_G.backward()
        optimizer_G.step()
        
        # Train Input Domain Discriminator
        optimizer_D1.zero_grad()

        loss_D1 = (
            compute_loss_D_adv(D1, X_1, valid)
            + compute_loss_D_adv(D1, X_21.detach(), fake)
        )

        loss_D1.backward()
        optimizer_D1.step()

        # Train Output Domain Discriminator
        optimizer_D2.zero_grad()

        loss_D2 = (
            compute_loss_D_adv(D2, X_2, valid)
            + compute_loss_D_adv(D2, X_12.detach(), fake)
        )
        loss_D2.backward()
        optimizer_D2.step()

        # Log
        batches_done = epoch * len(train_dataloader) + i

        if batches_done % opt.log_interval == 0:
            writer.add_scalar("00.Overall/01. Total Generator Loss", loss_G, batches_done)
            writer.add_scalar("00.Overall/02. Total Discriminator A->B Loss", loss_D1, batches_done)
            writer.add_scalar("00.Overall/03. Total Discriminator B->A Loss", loss_D2, batches_done)

            writer.add_scalar("01.Generator A->B/01. Image Domain Adversarial Loss A->B", loss_adv_2, batches_done)
            writer.add_scalar("01.Generator A->B/03. Common Feature Reconstruction Loss A->B", loss_com_rec_1, batches_done)
            writer.add_scalar("01.Generator A->B/05. Image Reconstruction Loss A->B", loss_img_rec_1, batches_done)
            writer.add_scalar("01.Generator A->B/06. Image Cycle Reconstruction Loss A->B", loss_img_cyc_1, batches_done)

            writer.add_scalar("02.Generator B->A/01. Image Domain Adversarial Loss B->A", loss_adv_1, batches_done)
            writer.add_scalar("02.Generator B->A/03. Common Feature Reconstruction Loss B->A", loss_com_rec_2, batches_done)
            writer.add_scalar("02.Generator B->A/05. Image Reconstruction Loss B->A", loss_img_rec_2, batches_done)
            writer.add_scalar("02.Generator B->A/06. Image Cycle Reconstruction Loss B->A", loss_img_cyc_2, batches_done)

            Enc1.eval()
            Dec1.eval()
            Enc2.eval()
            Dec2.eval()

            with torch.no_grad():
                batch = next(iter(val_dataloader))

                X_1 = Variable(batch["A"].type(Tensor)).to(device)
                X_2 = Variable(batch["B"].type(Tensor)).to(device)

                # Iniital feature maps
                Z_1 = Enc1(X_1)
                Z_2 = Enc2(X_2)

                # Reconstruction
                X_11 = Dec1(Z_1)
                X_22 = Dec2(Z_2)

                # Translate images to opposite domain
                X_12 = Dec2(Z_1)
                X_21 = Dec1(Z_2)

                # Translated feature maps
                Z_12 = Enc2(X_12)
                Z_21 = Enc1(X_21)

                # Cycle reconstruction
                X_121 = Dec1(Z_12)
                X_212 = Dec2(Z_21)

                X_1_grid = make_grid(X_1, nrow=5, normalize=True, value_range=(0, X_1.max()))
                X_2_grid = make_grid(X_2, nrow=5, normalize=True, value_range=(0, X_2.max()))

                X_11_grid = make_grid(X_11, nrow=5, normalize=True, value_range=(0, X_11.max()))
                X_22_grid = make_grid(X_22, nrow=5, normalize=True, value_range=(0, X_22.max()))
                
                X_12_grid = make_grid(X_12, nrow=5, normalize=True, value_range=(0, X_12.max()))
                X_21_grid = make_grid(X_21, nrow=5, normalize=True, value_range=(0, X_21.max()))

                X_121_grid = make_grid(X_121, nrow=5, normalize=True, value_range=(0, X_121.max()))
                X_212_grid = make_grid(X_212, nrow=5, normalize=True, value_range=(0, X_212.max()))

                writer.add_image("01. Cytosolic->NLS/01. Real Cytosolic Image (X_1)", X_1_grid, batches_done)
                for ch in range(Z_1.shape[1]):
                    Z_1_common_grid = make_grid(Z_1[:,ch,:,:], nrow=5, normalize=True)
                    writer.add_image(f"01. Cytosolic->NLS/02_{ch}. Common Feature CH {ch} (Z_1_common)", Z_1_common_grid, batches_done)
                writer.add_image("01. Cytosolic->NLS/03. Reconstructed Cytosolic Image (X_11)", X_11_grid, batches_done)
                writer.add_image("01. Cytosolic->NLS/04. Translated NLS Image (X_12)", X_12_grid, batches_done)
                for ch in range(Z_12.shape[1]):
                    Z_12_common_grid = make_grid(Z_12[:,ch,:,:], nrow=5, normalize=True)
                    writer.add_image(f"01. Cytosolic->NLS/05_{ch}. Translated Common Feature CH {ch} (Z_12_common)", Z_12_common_grid, batches_done)
                writer.add_image("01. Cytosolic->NLS/06. Cycle Reconstructed Cytosolic Image (X_121)", X_121_grid, batches_done)

                writer.add_image("02. NLS->Cytosolic/01. Real NLS Image (X_2)", X_2_grid, batches_done)
                for ch in range(Z_2.shape[1]):
                    Z_2_common_grid = make_grid(Z_2[:,ch,:,:], nrow=5, normalize=True)
                    writer.add_image(f"02. NLS->Cytosolic/02_{ch}. Common Feature CH {ch} (Z_2_common)", Z_2_common_grid, batches_done)
                writer.add_image("02. NLS->Cytosolic/03. Reconstructed NLS Image (X_22)", X_22_grid, batches_done)
                writer.add_image("02. NLS->Cytosolic/04. Translated Cytosolic Image (X_21)", X_21_grid, batches_done)
                for ch in range(Z_21.shape[1]):
                    Z_21_common_grid = make_grid(Z_21[:,ch,:,:], nrow=5, normalize=True)
                    writer.add_image(f"02. NLS->Cytosolic/05_{ch}. Translated Common Feature CH {ch} (Z_21_common)", Z_21_common_grid, batches_done)
                writer.add_image("02. NLS->Cytosolic/06. Cycle Reconstructed NLS Image (X_212)", X_212_grid, batches_done)
                
            writer.flush()


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D1.step()
    lr_scheduler_D2.step()
    
    # Save model and optimizer checkpoints
    if epoch % opt.checkpoint_interval == 0:
        torch.save(Enc1.state_dict(), f"{saved_models_dir}/Enc1_{epoch}.pth")
        torch.save(Dec1.state_dict(), f"{saved_models_dir}/Dec1_{epoch}.pth")
        torch.save(Enc2.state_dict(), f"{saved_models_dir}/Enc2_{epoch}.pth")
        torch.save(Dec2.state_dict(), f"{saved_models_dir}/Dec2_{epoch}.pth")
        torch.save(D1.state_dict(), f"{saved_models_dir}/D1_{epoch}.pth")
        torch.save(D2.state_dict(), f"{saved_models_dir}/D2_{epoch}.pth")
        torch.save(optimizer_G.state_dict(), f"{saved_models_dir}/optimizer_G_{epoch}.pth")
        torch.save(optimizer_D1.state_dict(), f"{saved_models_dir}/optimizer_D1_{epoch}.pth")
        torch.save(optimizer_D2.state_dict(), f"{saved_models_dir}/optimizer_D2_{epoch}.pth")
        torch.save(lr_scheduler_G.state_dict(), f"{saved_models_dir}/lr_scheduler_G_{epoch}.pth")
        torch.save(lr_scheduler_D1.state_dict(), f"{saved_models_dir}/lr_scheduler_D1_{epoch}.pth")
        torch.save(lr_scheduler_D2.state_dict(), f"{saved_models_dir}/lr_scheduler_D2_{epoch}.pth")
