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
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from datasets import *
from utils import *
from models import UNet, MultiDiscriminator

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()

# Training parameters
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100000000, help="number of epochs of training")
parser.add_argument("--iter_per_epoch", type=int, default=1, help="number of iterations per epoch")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")

parser.add_argument("--lr_G", type=float, default=3e-4, help="adam: learning rate")
parser.add_argument("--lr_D_1", type=float, default=3e-4, help="adam: learning rate")
parser.add_argument("--lr_D_2", type=float, default=3e-4, help="adam: learning rate")

parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--gpu_ids", type=int, default=[0], nargs="+", help="gpu ids to use")
parser.add_argument("--seed", type=int, help="seed, None if you want a random seed")
parser.add_argument("--momentum", type=float, default=0.1, help="momentum for batch normalization")

# Logging parameters
parser.add_argument("--exp_name", type=str, default=None, help="name of the experiment")
parser.add_argument("--exp_tag", type=str, default=None, help="tag for the experiment")
parser.add_argument("--output_dir", type=str, default="./results", help="output directory")
parser.add_argument("--log_loss_interval", type=int, default=5, help="interval saving generator samples")
parser.add_argument("--log_image_interval", type=int, default=10, help="interval saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
parser.add_argument("--val_interval", type=int, default=1, help="interval between validation")

# Dataset parameters
parser.add_argument("--dataset_dir", type=str, default="./")
parser.add_argument("--data_type", type=str, default="c2n", help="[c2n | stain]", choices=['c2n', 'stain'])
parser.add_argument("--crop_size", type=int, default=256, help="Crop size")
parser.add_argument("--augmentation", type=int, default=1, help="augment data")
parser.add_argument("--normalize", type=int, default=1, help="normalize data")
parser.add_argument("--scale_ratio", type=float, default=[1.0, 1.0], nargs="+", help="scale ratio")
parser.add_argument("--shuffle", type=int, default=1, help="shuffle data")
parser.add_argument("--unpaired", type=int, default=1, help="unpaired data")

# Generator parameters
parser.add_argument("--n_ch_in", type=int, default=3, help="number of channels in the input image")
parser.add_argument("--n_ch_out", type=int, default=3, help="number of channels in the output image")
parser.add_argument("--n_ch_com", type=int, default=3, help="number of channels in the common feature")
parser.add_argument("--enc_act", type=str, default="relu", help="[none | sigmoid | tanh | softmax]", choices=['sigmoid', 'tanh', 'softmax', 'leakyrelu', 'relu'])
parser.add_argument("--dec_act", type=str, default="relu", help="[none | sigmoid | tanh | softmax]", choices=['sigmoid', 'tanh', 'softmax', 'leakyrelu', 'relu'])
parser.add_argument("--G_mid_ch", nargs="+", type=int, default=[64, 128, 256, 512, 1024], help="number of channels in the middle of the unet")
parser.add_argument("--G_demodulated", type=int, default=0)
parser.add_argument("--G_norm_type", type=str, default="batch", help="[batch | instance | none]", choices=['batch', 'instance', 'none'])

# Discriminator parameters
parser.add_argument("--D_n_scales", type=int, default=3, help="number of scales for the multi-scale discriminator")
parser.add_argument("--D_downsample_stride", type=int, default=2, help="stride for downsample for the multi-scale discriminator (ex: 2=1/2, 4=1/4)")
parser.add_argument("--D_norm_type", type=str, default="instance", help="[batch | instance ]", choices=['batch', 'instance', 'none'])
parser.add_argument("--D_n_layers", type=int, default=3, help="number of layers in the discriminator")

# Loss parameters
parser.add_argument("--lambda_img_adv_1", type=float, default=0, help="weight for image domain adversarial loss")
parser.add_argument("--lambda_img_adv_2", type=float, default=0, help="weight for image domain adversarial loss")
parser.add_argument("--lambda_com_rec_1", type=float, default=0, help="weight for common feature reconstruction loss")
parser.add_argument("--lambda_com_rec_2", type=float, default=0, help="weight for common feature reconstruction loss")
parser.add_argument("--lambda_img_cyc_1", type=float, default=0, help="weight for image cycle reconstruction loss")
parser.add_argument("--lambda_img_cyc_2", type=float, default=0, help="weight for image cycle reconstruction loss")

parser.add_argument("--lambda_img_cyc_exp_factor", type=float, default=1.00, help="exponential factor for image cycle loss")
parser.add_argument("--lambda_img_cyc_max_weight", type=float, default=10, help="maximum weight for image cycle loss")


opt = parser.parse_args()
print(opt)

# Fix Seed
if opt.seed != None:
    print(f"Seed: {opt.seed}")
else:
    opt.seed = np.random.randint(1, 10000)
    print(f"Random Seed: {opt.seed}")

if opt.exp_name == None:
    from datetime import date
    today = date.today().strftime("%d%m%Y")
    opt.exp_name = f"{today}_seed_{opt.seed}_{os.path.basename(opt.dataset_dir)}_{opt.crop_size}_zch_{opt.n_ch_com}"

if opt.exp_tag != None:
    opt.exp_name = f"{opt.exp_name}_{opt.exp_tag}"

print(opt.exp_name)

np.random.seed(opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

cuda = torch.cuda.is_available()
if cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

options_dict = vars(opt)

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

Enc1 = UNet(n_in=opt.n_ch_in, n_out=opt.n_ch_com, mid_channels=opt.G_mid_ch, norm_type=opt.G_norm_type, demodulated=opt.G_demodulated, act=opt.enc_act, momentum=opt.momentum)
Dec1 = UNet(n_in=opt.n_ch_com, n_out=opt.n_ch_in, mid_channels=opt.G_mid_ch, norm_type=opt.G_norm_type, demodulated=opt.G_demodulated, act=opt.dec_act, momentum=opt.momentum)
Enc2 = UNet(n_in=opt.n_ch_out, n_out=opt.n_ch_com, mid_channels=opt.G_mid_ch, norm_type=opt.G_norm_type, demodulated=opt.G_demodulated, act=opt.enc_act,  momentum=opt.momentum)
Dec2 = UNet(n_in=opt.n_ch_com, n_out=opt.n_ch_out, mid_channels=opt.G_mid_ch, norm_type=opt.G_norm_type, demodulated=opt.G_demodulated, act=opt.dec_act, momentum=opt.momentum)

D1 = MultiDiscriminator(channels=opt.n_ch_in, num_scales=opt.D_n_scales, num_layers=opt.D_n_layers, downsample_stride=opt.D_downsample_stride, norm_type=opt.D_norm_type, kernel_size=4, stride=2, padding=1)
D2 = MultiDiscriminator(channels=opt.n_ch_out, num_scales=opt.D_n_scales, num_layers=opt.D_n_layers, downsample_stride=opt.D_downsample_stride, norm_type=opt.D_norm_type, kernel_size=4, stride=2, padding=1)

criterion_recon_cyc = nn.L1Loss()
criterion_recon_com = nn.L1Loss()

if len(opt.gpu_ids) > 1:
    # use multiple GPUs via DataParallel
    print(f"Using {len(opt.gpu_ids)} GPUs")
    Enc1 = nn.DataParallel(Enc1, device_ids=opt.gpu_ids)
    Dec1 = nn.DataParallel(Dec1, device_ids=opt.gpu_ids)
    Enc2 = nn.DataParallel(Enc2, device_ids=opt.gpu_ids)
    Dec2 = nn.DataParallel(Dec2, device_ids=opt.gpu_ids)
    D1 = nn.DataParallel(D1, device_ids=opt.gpu_ids)
    D2 = nn.DataParallel(D2, device_ids=opt.gpu_ids)
    device = torch.device("cuda" if cuda else "cpu")
else:
    device = torch.device(f"cuda:{opt.gpu_ids[0]}" if cuda else "cpu")
if cuda:
    Enc1.to(device)
    Dec1.to(device)
    Enc2.to(device)
    Dec2.to(device)
    D1.to(device)
    D2.to(device)
    criterion_recon_cyc.to(device)
    criterion_recon_com.to(device)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

optimizer_G = torch.optim.AdamW(
    itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters()),
    lr=opt.lr_G
)
optimizer_D1 = torch.optim.AdamW(D1.parameters(), lr=opt.lr_D_1)
optimizer_D2 = torch.optim.AdamW(D2.parameters(), lr=opt.lr_D_2)


if opt.epoch != 0:
    Enc1.load_state_dict(torch.load(f"{saved_models_dir}/Enc1_{opt.epoch}.pth"))
    Dec1.load_state_dict(torch.load(f"{saved_models_dir}/Dec1_{opt.epoch}.pth"))
    Enc2.load_state_dict(torch.load(f"{saved_models_dir}/Enc2_{opt.epoch}.pth"))
    Dec2.load_state_dict(torch.load(f"{saved_models_dir}/Dec2_{opt.epoch}.pth"))
    D1.load_state_dict(torch.load(f"{saved_models_dir}/D1_{opt.epoch}.pth"))
    D2.load_state_dict(torch.load(f"{saved_models_dir}/D2_{opt.epoch}.pth"))
    optimizer_G.load_state_dict(torch.load(f"{saved_models_dir}/optimizer_G_{opt.epoch}.pth"))
    optimizer_D1.load_state_dict(torch.load(f"{saved_models_dir}/optimizer_D1_{opt.epoch}.pth"))
    optimizer_D2.load_state_dict(torch.load(f"{saved_models_dir}/optimizer_D2_{opt.epoch}.pth"))

opt.scale_ratio = (opt.scale_ratio[0], opt.scale_ratio[1])

if opt.data_type == "c2n":
    train_dataloader = DataLoader(
        ImageDataset(base_dataset_dir=opt.dataset_dir, mode="train", normalize=opt.normalize, seed=opt.seed, size=opt.crop_size, augmentation=opt.augmentation, scale_ratio=opt.scale_ratio),
        batch_size=opt.batch_size, num_workers=opt.n_cpu, shuffle=opt.shuffle, drop_last=True
    )
    val_dataloader = DataLoader(
        ImageDataset(base_dataset_dir=opt.dataset_dir, mode="test", normalize=opt.normalize, seed=opt.seed, size=opt.crop_size, augmentation=False, scale_ratio=opt.scale_ratio),
        batch_size=1, num_workers=1, shuffle=False, drop_last=True
    )
elif opt.data_type == "stain":
    train_dataloader = DataLoader(
        HnEDataset(base_dataset_dir=opt.dataset_dir, mode="train", normalize=opt.normalize, seed=opt.seed, size=opt.crop_size, augmentation=opt.augmentation, scale_ratio=opt.scale_ratio, shuffle=opt.unpaired),
        batch_size=opt.batch_size, num_workers=opt.n_cpu, shuffle=opt.shuffle, drop_last=True
    )
    val_dataloader = DataLoader(
        HnEDataset(base_dataset_dir=opt.dataset_dir, mode="test", normalize=opt.normalize, seed=opt.seed, size=opt.crop_size, augmentation=False, scale_ratio=opt.scale_ratio),
        batch_size=8, num_workers=1, shuffle=False, drop_last=True
    )

print("Length of Train Dataloader: ", len(train_dataloader))
print("Length of Val Dataloader: ", len(val_dataloader))

# Adversarial ground truths
valid = 1
fake = 0

# Train loop
batches_done = 0
iters_done = 0
for epoch in range(opt.epoch, opt.n_epochs):
    for e_i in tqdm(range(opt.iter_per_epoch), desc=f"Training Epoch {epoch}", position=0):
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Training Iteration {e_i}", position=1, leave=False)):
          
            Enc1.train()
            Dec1.train()
            Enc2.train()
            Dec2.train()

            # Input data
            X_1 = Variable(batch["A"].type(Tensor)).to(device)
            X_2 = Variable(batch["B"].type(Tensor)).to(device)

            Z_1 = Enc1(X_1)
            Z_2 = Enc2(X_2)
            
            # Translate images to opposite domain
            X_12 = Dec2(Z_1)
            X_21 = Dec1(Z_2)

            Z_12 = Enc2(X_12)
            Z_21 = Enc1(X_21)

            # Cycle reconstruction
            X_121 = Dec1(Z_12)
            X_212 = Dec2(Z_21)
                
            # Total Loss
            loss_G = 0.0

            # Image Domain Adversarial Losses
            loss_adv_1 = D1.compute_loss(D1, X_21, valid)
            loss_G += opt.lambda_img_adv_1 * loss_adv_1

            loss_adv_2 = D2.compute_loss(D2, X_12, valid)
            loss_G += opt.lambda_img_adv_2 * loss_adv_2

            # Common feature map reconstruction loss
            loss_com_rec_1 = criterion_recon_com(Z_12, Z_1.detach())
            loss_G += opt.lambda_com_rec_1 * loss_com_rec_1

            loss_com_rec_2 = criterion_recon_com(Z_21.detach(), Z_2)
            loss_G += opt.lambda_com_rec_2 * loss_com_rec_2

            # Image cycle reconstruction loss
            exp_lambda_img_cyc_1 = min(opt.lambda_img_cyc_1 * (opt.lambda_img_cyc_exp_factor ** iters_done), opt.lambda_img_cyc_max_weight)
            loss_img_cyc_1 = criterion_recon_cyc(X_121, X_1)
            loss_G += exp_lambda_img_cyc_1 * loss_img_cyc_1

            exp_lambda_img_cyc_2 = min(opt.lambda_img_cyc_2 * (opt.lambda_img_cyc_exp_factor ** iters_done), opt.lambda_img_cyc_max_weight)
            loss_img_cyc_2 = criterion_recon_cyc(X_212, X_2)
            loss_G += exp_lambda_img_cyc_2 * loss_img_cyc_2

            # Compute Losses
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
    
            # Train Input Domain Discriminator
            optimizer_D1.zero_grad()
            loss_D1 = (
                D1.compute_loss(D1, X_1, valid)
                + D1.compute_loss(D1, X_21.detach(), fake)
            )
            loss_D1.backward()
            optimizer_D1.step()

            # Train Output Domain Discriminator
            optimizer_D2.zero_grad()
            loss_D2 = (
                D2.compute_loss(D2, X_2, valid)
                + D2.compute_loss(D2, X_12.detach(), fake)
            )
            loss_D2.backward()
            optimizer_D2.step()

            if (batches_done + 1) % opt.log_loss_interval == 0:
                writer.add_scalar("00.Overall (Train)/01. Total Generator Loss", loss_G, batches_done)
                writer.add_scalar("00.Overall (Train)/02. Total Discriminator A Loss", loss_D1, batches_done)
                writer.add_scalar("00.Overall (Train)/03. Total Discriminator B Loss", loss_D2, batches_done)

                writer.add_scalar("01.Generator A->B (Train)/01. Image Domain Adversarial Loss A->B", loss_adv_2, batches_done)
                writer.add_scalar("01.Generator A->B (Train)/02. Common Feature Reconstruction Loss A->B", loss_com_rec_1, batches_done)
                writer.add_scalar("01.Generator A->B (Train)/03. Image Cycle Reconstruction Loss A->B", loss_img_cyc_1, batches_done)

                writer.add_scalar("02.Generator B->A (Train)/01. Image Domain Adversarial Loss B->A", loss_adv_1, batches_done)
                writer.add_scalar("02.Generator B->A (Train)/02. Common Feature Reconstruction Loss B->A", loss_com_rec_2, batches_done)
                writer.add_scalar("02.Generator B->A (Train)/03. Image Cycle Reconstruction Loss B->A", loss_img_cyc_2, batches_done)

            if (batches_done + 1) % opt.log_image_interval == 0:

                X_1_grid = make_grid(X_1[0], normalize=True)
                X_2_grid = make_grid(X_2[0], normalize=True)                
                X_12_grid = make_grid(X_12[0], normalize=True)
                X_21_grid = make_grid(X_21[0], normalize=True)
                X_121_grid = make_grid(X_121[0], normalize=True)
                X_212_grid = make_grid(X_212[0], normalize=True)

                writer.add_image("01. Cytosolic->NLS (Train)/01. Real Cytosolic Image (X_1)", X_1_grid, batches_done)
                for ch in range(Z_1.shape[1]):
                    Z_1_common_grid = make_grid(Z_1[0,ch,:,:], normalize=True)
                    writer.add_image(f"01. Cytosolic->NLS (Train) Z Channels 1 /02_{ch}. Common Feature CH {ch} (Z_1_common)", Z_1_common_grid, batches_done)
                writer.add_image("01. Cytosolic->NLS (Train)/03. Translated NLS Image (X_12)", X_12_grid, batches_done)
                for ch in range(Z_12.shape[1]):
                    Z_12_common_grid = make_grid(Z_12[0,ch,:,:], normalize=True)
                    writer.add_image(f"01. Cytosolic->NLS (Train) Z Channels 2 /04_{ch}. Translated Common Feature CH {ch} (Z_12_common)", Z_12_common_grid, batches_done)
                writer.add_image("01. Cytosolic->NLS (Train)/05. Cycle Reconstructed Cytosolic Image (X_121)", X_121_grid, batches_done)

                writer.add_image("02. NLS->Cytosolic (Train)/01. Real NLS Image (X_2)", X_2_grid, batches_done)
                for ch in range(Z_2.shape[1]):
                    Z_2_common_grid = make_grid(Z_2[0,ch,:,:], normalize=True)
                    writer.add_image(f"02. NLS->Cytosolic Z Channels 1 (Train)/02_{ch}. Common Feature CH {ch} (Z_2_common)", Z_2_common_grid, batches_done)
                writer.add_image("02. NLS->Cytosolic (Train)/04. Translated Cytosolic Image (X_21)", X_21_grid, batches_done)
                for ch in range(Z_21.shape[1]):
                    Z_21_common_grid = make_grid(Z_21[0,ch,:,:], normalize=True)
                    writer.add_image(f"02. NLS->Cytosolic Z Channels 2 (Train)/04_{ch}. Translated Common Feature CH {ch} (Z_21_common)", Z_21_common_grid, batches_done)
                writer.add_image("02. NLS->Cytosolic (Train)/05. Cycle Reconstructed NLS Image (X_212)", X_212_grid, batches_done)
                
                if opt.n_ch_in == 1 and opt.n_ch_out == 1:
                    X_1_overlay = overlay_images(X_12[0], X_1[0])
                    X_1_overlay_grid = make_grid(X_1_overlay, normalize=True)

                    X_2_overlay = overlay_images(X_21[0], X_2[0])
                    X_2_overlay_grid = make_grid(X_2_overlay, normalize=True)

                    writer.add_image("01. Cytosolic->NLS (Train)/00. Overlay Cytosolic Image (X_12)", X_1_overlay_grid, batches_done)
                    writer.add_image("02. NLS->Cytosolic (Train)/00. Overlay NLS Image (X_21)", X_2_overlay_grid, batches_done)

            batches_done += 1
                
        iters_done += 1
    
    # Validation
    Enc1.eval()
    Dec1.eval()
    Enc2.eval()
    Dec2.eval()

    if epoch % opt.val_interval == 0:
        with torch.no_grad():
            batch = next(iter(val_dataloader))

            X_1 = Variable(batch["A"].type(Tensor)).to(device)
            X_2 = Variable(batch["B"].type(Tensor)).to(device)

            Z_1 = Enc1(X_1)
            Z_2 = Enc2(X_2)
            
            # Translate images to opposite domain
            X_12 = Dec2(Z_1)
            X_21 = Dec1(Z_2)

            Z_12 = Enc2(X_12)
            Z_21 = Enc1(X_21)

            # Cycle reconstruction
            X_121 = Dec1(Z_12)
            X_212 = Dec2(Z_21)

            X_1_grid = make_grid(X_1, normalize=True)
            X_2_grid = make_grid(X_2, normalize=True)
            
            X_12_grid = make_grid(X_12, normalize=True)
            X_21_grid = make_grid(X_21, normalize=True)

            X_121_grid = make_grid(X_121, normalize=True)
            X_212_grid = make_grid(X_212, normalize=True)

            writer.add_image("01. Cytosolic->NLS (Val)/01. Real Cytosolic Image (X_1)", X_1_grid, epoch)
            for ch in range(Z_1.shape[1]):
                Z_1_common_grid = make_grid(Z_1[:,ch:ch+1,:,:], normalize=True)
                writer.add_image(f"01. Cytosolic->NLS (Val) Feature/02_{ch}. Common Feature CH {ch} (Z_1_common)", Z_1_common_grid, epoch)
            writer.add_image("01. Cytosolic->NLS (Val)/03. Translated NLS Image (X_12)", X_12_grid, epoch)
            for ch in range(Z_12.shape[1]):
                Z_12_common_grid = make_grid(Z_12[:,ch:ch+1,:,:], normalize=True)
                writer.add_image(f"01. Cytosolic->NLS (Val) Feature/04_{ch}. Translated Common Feature CH {ch} (Z_12_common)", Z_12_common_grid, epoch)
            writer.add_image("01. Cytosolic->NLS (Val)/05. Cycle Reconstructed Cytosolic Image (X_121)", X_121_grid, epoch)

            writer.add_image("02. NLS->Cytosolic (Val)/01. Real NLS Image (X_2)", X_2_grid, epoch)
            for ch in range(Z_2.shape[1]):
                Z_2_common_grid = make_grid(Z_2[:,ch:ch+1,:,:], normalize=True)
                writer.add_image(f"02. NLS->Cytosolic (Val) Feature/02_{ch}. Common Feature CH {ch} (Z_2_common)", Z_2_common_grid, epoch)
            writer.add_image("02. NLS->Cytosolic (Val)/03. Translated Cytosolic Image (X_21)", X_21_grid, epoch)
            for ch in range(Z_21.shape[1]):
                Z_21_common_grid = make_grid(Z_21[:,ch:ch+1,:,:], normalize=True)
                writer.add_image(f"02. NLS->Cytosolic (Val) Feature/04_{ch}. Translated Common Feature CH {ch} (Z_21_common)", Z_21_common_grid, epoch)
            writer.add_image("02. NLS->Cytosolic (Val)/05. Cycle Reconstructed NLS Image (X_212)", X_212_grid, epoch)
            
            if opt.n_ch_in == 1 and opt.n_ch_out == 1:
                X_1_overlay = overlay_images(X_12[0], X_1[0])
                X_1_overlay_grid = make_grid(X_1_overlay, normalize=True)

                X_2_overlay = overlay_images(X_21[0], X_2[0])
                X_2_overlay_grid = make_grid(X_2_overlay, normalize=True)

                writer.add_image("01. Cytosolic->NLS (Val)/00. Overlay Cytosolic Image (X_12)", X_1_overlay_grid, epoch)
                writer.add_image("02. NLS->Cytosolic (Val)/00. Overlay NLS Image (X_21)", X_2_overlay_grid, epoch)
            
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
