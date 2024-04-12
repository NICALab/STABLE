import argparse
import os
import numpy as np
import itertools
import json
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from models import *
from datasets import *
import torch
import warnings
import random
import PIL
from unet2d import UNet2D
from utils import overlay_images, random_rotate, random_flip
from cyclegan.models import Discriminator
import torchvision.transforms as transforms
import random

from unet import UNet

warnings.filterwarnings("ignore", category=UserWarning)

PIL.Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser()

# Training parameters
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100000000, help="number of epochs of training")
parser.add_argument("--iter_per_epoch", type=int, default=10, help="number of iterations per epoch")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")

parser.add_argument("--lr_G", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--lr_D_1", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--lr_D_2", type=float, default=0.0001, help="adam: learning rate")

parser.add_argument("--D_delay", type=int, default=0, help="number of iterations to delay the learning the discriminator")

parser.add_argument("--weight_decay", type=float, default=0, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--step_size", type=int, default=1000, help="adam: decay of first order momentum of gradient")
parser.add_argument("--gamma", type=float, default=0.5, help="how much to decay learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# parser.add_argument("--gpu_id", type=int, default=1, help="gpu id to train on")
parser.add_argument("--gpu_ids", type=int, default=[0], nargs="+", help="gpu ids to use")
parser.add_argument("--seed", type=int, help="seed, None if you want a random seed")

parser.add_argument("--clip_mode", type=str, default="norm", help="[norm | value] clip mode for weights")
parser.add_argument("--clip_value", type=float, default=10, help="clip value for weights")

# Logging parameters
parser.add_argument("--exp_name", type=str, default="", help="name of the experiment")
parser.add_argument("--output_dir", type=str, default="./results", help="output directory")

parser.add_argument("--log_loss_interval", type=int, default=5, help="interval saving generator samples")
parser.add_argument("--log_image_interval", type=int, default=10, help="interval saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")
parser.add_argument("--val_interval", type=int, default=1, help="interval between validation")

# Dataset parameters
parser.add_argument("--dataset_dir", type=str, default="/media/HDD2/Josh/Cytosolic2NLS/Data/230427_Live_to_HnE")
parser.add_argument("--datatype", type=str, default="tif", help="[tif | png]")
parser.add_argument("--crop_size", type=int, default=256, help="Crop size")
parser.add_argument("--augmentation", type=int, default=1, help="augment data")
# parser.add_argument("--normalize", type=str, default="dataset", help="[ dataset | data ] normalize data against entire dataset = dataset, or normalize each image = data")
parser.add_argument("--normalize", type=int, default=1, help="normalize data")
parser.add_argument("--load_to_memory", type=int, default=0, help="load data to memory")
parser.add_argument("--equalize", type=int, default=[0, 0], nargs="+", help="gpu ids to use")
parser.add_argument("--scale_ratio", type=float, default=[1.0, 1.0], nargs="+", help="scale ratio")

# Model parameters
parser.add_argument("--num_scales", type=int, default=3, help="number of scales for the multi-scale discriminator")
parser.add_argument("--downsample_stride", type=int, default=2, help="stride for downsample for the multi-scale discriminator (ex: 2=1/2, 4=1/4)")
parser.add_argument("--input_channels", type=int, default=3, help="number of channels in the input image")
parser.add_argument("--output_channels", type=int, default=3, help="number of channels in the output image")
parser.add_argument("--common_channels", type=int, default=3, help="number of channels in the common feature")
parser.add_argument("--std", type=int, default=0, help="std")
parser.add_argument("--conv_type", type=str, default="Conv2d", help="[Conv2d | DemodulatedConv2d]")
parser.add_argument("--model_norm", type=int, default=0, help="normalize the model")

parser.add_argument("--unet_mid_channels", nargs="+", type=int, default=[64, 128, 256, 512, 1024], help="number of channels in the middle of the unet")
parser.add_argument("--unet_final_channels", nargs="+", type=int, default=[32,16], help="number of channels in the final layer of the unet")
parser.add_argument("--unet_norm_type", type=str, default="instance", help="[batch | instance]")
parser.add_argument("--unet_skip_comb", type=str, default="concat", help="[concat | add]")
parser.add_argument("--unet_up_method", type=str, default="UpsampleConv", help="[UpsampleConv | ConvTranspose]")
parser.add_argument("--unet_conv_method", type=str, default="conv2d", help="[conv2d | demodconv2d]")
parser.add_argument("--unet_n_conv", type=int, default=2, help="number of convolutions in each block")

parser.add_argument("--kl_reg", type=int, default=0, help="kl regularization")
parser.add_argument("--mu_dim", type=int, default=64, help="mu dimension")

parser.add_argument("--discriminator_type", type=str, default="patch", help="[multi | patch]")

# Loss parameters
parser.add_argument("--lambda_img_adv_1", type=float, default=10, help="weight for image domain adversarial loss")
parser.add_argument("--lambda_img_adv_2", type=float, default=10, help="weight for image domain adversarial loss")
parser.add_argument("--lambda_com_rec_1", type=float, default=1, help="weight for common feature reconstruction loss")
parser.add_argument("--lambda_com_rec_2", type=float, default=1, help="weight for common feature reconstruction loss")
parser.add_argument("--lambda_img_rec_1", type=float, default=100, help="weight for image reconstruction loss")
parser.add_argument("--lambda_img_rec_2", type=float, default=100, help="weight for image reconstruction loss")
parser.add_argument("--lambda_img_cyc_1", type=float, default=0, help="weight for image cycle reconstruction loss")
parser.add_argument("--lambda_img_cyc_2", type=float, default=0, help="weight for image cycle reconstruction loss")
parser.add_argument("--lambda_kl_1", type=float, default=0, help="weight for kl regularization")
parser.add_argument("--lambda_kl_2", type=float, default=0, help="weight for kl regularization")
parser.add_argument("--lambda_aug", type=float, default=0, help="weight for data augmentation")
parser.add_argument("--lambda_l1_1", type=float, default=0, help="weight for l1 regularization")
parser.add_argument("--lambda_l1_2", type=float, default=0, help="weight for l1 regularization")

opt = parser.parse_args()
print(opt)

# Fix Seed
if opt.seed != None:
    print(f"Seed: {opt.seed}")
else:
    opt.seed = np.random.randint(1, 10000)
    print(f"Random Seed: {opt.seed}")

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
if torch.cuda.is_available():
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

# Enc1 = Unet(input_channels=opt.input_channels, output_channels=opt.common_channels, upsampling_method="InterpolationConv", std=opt.std, norm=opt.model_norm, conv_type=opt.conv_type)
# Dec1 = Unet(input_channels=opt.common_channels, output_channels=opt.input_channels, upsampling_method="InterpolationConv", norm=opt.model_norm, conv_type=opt.conv_type)
# Enc2 = Unet(input_channels=opt.output_channels, output_channels=opt.common_channels, upsampling_method="InterpolationConv", std=opt.std, norm=opt.model_norm, conv_type=opt.conv_type)
# Dec2 = Unet(input_channels=opt.common_channels, output_channels=opt.output_channels, upsampling_method="InterpolationConv", norm=opt.model_norm, conv_type=opt.conv_type)

if opt.lambda_kl_1 or opt.lambda_kl_2 > 0:
    kl_reg_1 = True
    kl_reg_2 = True
else:
    kl_reg_1 = False
    kl_reg_2 = False
# Enc1 = UNet2D(in_channels=opt.input_channels, out_channels=opt.common_channels, mid_channels=opt.unet_mid_channels, final_channels=opt.unet_final_channels, kernel_size=3, stride=1, padding=1, norm_type=opt.unet_norm_type, skip_comb=opt.unet_skip_comb, up_method=opt.unet_up_method, conv_method=opt.unet_conv_method, n_conv=opt.unet_n_conv, kl_reg=kl_reg_1, mu_dim=opt.mu_dim)
# Dec1 = UNet2D(in_channels=opt.common_channels, out_channels=opt.input_channels, mid_channels=opt.unet_mid_channels, final_channels=opt.unet_final_channels, kernel_size=3, stride=1, padding=1, norm_type=opt.unet_norm_type, skip_comb=opt.unet_skip_comb, up_method=opt.unet_up_method, conv_method=opt.unet_conv_method, n_conv=opt.unet_n_conv, kl_reg=False)
# Enc2 = UNet2D(in_channels=opt.output_channels, out_channels=opt.common_channels, mid_channels=opt.unet_mid_channels, final_channels=opt.unet_final_channels, kernel_size=3, stride=1, padding=1, norm_type=opt.unet_norm_type, skip_comb=opt.unet_skip_comb, up_method=opt.unet_up_method, conv_method=opt.unet_conv_method, n_conv=opt.unet_n_conv, kl_reg=kl_reg_2, mu_dim=opt.mu_dim)
# Dec2 = UNet2D(in_channels=opt.common_channels, out_channels=opt.output_channels, mid_channels=opt.unet_mid_channels, final_channels=opt.unet_final_channels, kernel_size=3, stride=1, padding=1, norm_type=opt.unet_norm_type, skip_comb=opt.unet_skip_comb, up_method=opt.unet_up_method, conv_method=opt.unet_conv_method, n_conv=opt.unet_n_conv, kl_reg=False)

if opt.conv_type == "DemodulatedConv2d":
    demodulated = True
elif opt.conv_type == "Conv2d":
    demodulated = False

Enc1 = UNet(n_channels=opt.input_channels, n_classes=opt.common_channels, mid_channels=opt.unet_mid_channels, bilinear=True, kl_reg=kl_reg_1, norm_type=opt.unet_norm_type, demodulated=demodulated)
Dec1 = UNet(n_channels=opt.common_channels, n_classes=opt.input_channels, mid_channels=opt.unet_mid_channels, bilinear=True, kl_reg=False, norm_type=opt.unet_norm_type, demodulated=demodulated)
Enc2 = UNet(n_channels=opt.output_channels, n_classes=opt.common_channels, mid_channels=opt.unet_mid_channels, bilinear=True, kl_reg=kl_reg_1, norm_type=opt.unet_norm_type, demodulated=demodulated)
Dec2 = UNet(n_channels=opt.common_channels, n_classes=opt.output_channels, mid_channels=opt.unet_mid_channels, bilinear=True, kl_reg=False, norm_type=opt.unet_norm_type, demodulated=demodulated)

if opt.discriminator_type == "multi":
    D1 = MultiDiscriminator(channels=opt.input_channels, num_scales=opt.num_scales, downsample_stride=opt.downsample_stride)
    D2 = MultiDiscriminator(channels=opt.output_channels, num_scales=opt.num_scales, downsample_stride=opt.downsample_stride)
elif opt.discriminator_type == "patch":
    input_shape = (opt.input_channels, opt.crop_size, opt.crop_size)
    D1 = Discriminator(input_shape)
    D2 = Discriminator(input_shape)

# from torchsummary import summary
# summary(Enc1, (opt.input_channels, opt.crop_size, opt.crop_size), device="cpu")
# summary(Dec1, (opt.common_channels, opt.crop_size, opt.crop_size), device="cpu")
# summary(D1, (opt.input_channels, opt.crop_size, opt.crop_size), device="cpu")
# exit()

criterion_recon = torch.nn.L1Loss()

cuda = torch.cuda.is_available()
if len(opt.gpu_ids) > 1:
    # use multiple GPUs via DataParallel
    print(f"Using {len(opt.gpu_ids)} GPUs")
    Enc1 = torch.nn.DataParallel(Enc1, device_ids=opt.gpu_ids)
    Dec1 = torch.nn.DataParallel(Dec1, device_ids=opt.gpu_ids)
    Enc2 = torch.nn.DataParallel(Enc2, device_ids=opt.gpu_ids)
    Dec2 = torch.nn.DataParallel(Dec2, device_ids=opt.gpu_ids)
    D1 = torch.nn.DataParallel(D1, device_ids=opt.gpu_ids)
    D2 = torch.nn.DataParallel(D2, device_ids=opt.gpu_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(f"cuda:{opt.gpu_ids[0]}")
if cuda:
    Enc1.to(device)
    Dec1.to(device)
    Enc2.to(device)
    Dec2.to(device)
    D1.to(device)
    D2.to(device)
print(device)
# Optimizers
# optimizer_G = torch.optim.Adam(
#     itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters()),
#     lr=opt.lr_G,
#     betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay
# )
# optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr_D_1, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
# optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr_D_2, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

# Learning rate update schedulers
# lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=opt.step_size,
#                                         gamma=opt.gamma, last_epoch=-1)
# lr_scheduler_D1 = torch.optim.lr_scheduler.StepLR(optimizer_D1, step_size=opt.step_size,
#                                         gamma=opt.gamma, last_epoch=-1)
# lr_scheduler_D2 = torch.optim.lr_scheduler.StepLR(optimizer_D2, step_size=opt.step_size,
#                                         gamma=opt.gamma, last_epoch=-1)

# Optimizers (AdamW)
# optimizer_G = torch.optim.AdamW(
#     itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters())
# )
# optimizer_D1 = torch.optim.AdamW(D1.parameters())
# optimizer_D2 = torch.optim.AdamW(D2.parameters())


optimizer_G = torch.optim.AdamW(
    itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters()),
    lr=opt.lr_G
)
optimizer_D1 = torch.optim.AdamW(D1.parameters(), lr=opt.lr_D_1)
optimizer_D2 = torch.optim.AdamW(D2.parameters(), lr=opt.lr_D_2)

# Learning rate update schedulers (CosineAnnealingLR)
# lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=500, eta_min=0, last_epoch=-1)
# lr_scheduler_D1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D1, T_max=500, eta_min=0, last_epoch=-1)
# lr_scheduler_D2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D2, T_max=500, eta_min=0, last_epoch=-1)

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
    # lr_scheduler_G.load_state_dict(torch.load(f"{saved_models_dir}/lr_scheduler_G_{opt.epoch}.pth"))
    # lr_scheduler_D1.load_state_dict(torch.load(f"{saved_models_dir}/lr_scheduler_D1_{opt.epoch}.pth"))
    # lr_scheduler_D2.load_state_dict(torch.load(f"{saved_models_dir}/lr_scheduler_D2_{opt.epoch}.pth"))
# else:
    # Initialize weights
    # Enc1.apply(weights_init('kaiming'))
    # Dec1.apply(weights_init('kaiming'))
    # Enc2.apply(weights_init('kaiming'))
    # Dec2.apply(weights_init('kaiming'))
    # D1.apply(weights_init('gaussian'))
    # D2.apply(weights_init('gaussian'))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

equalize = (opt.equalize[0], opt.equalize[1])
opt.scale_ratio = (opt.scale_ratio[0], opt.scale_ratio[1])
# Dataset loader
train_dataloader = DataLoader(
    ImageDataset(base_dataset_dir=opt.dataset_dir, mode="train", normalize=opt.normalize, datatype=opt.datatype, seed=opt.seed, size=opt.crop_size, test_idx=None, augmentation=opt.augmentation, equalize=equalize, scale_ratio=opt.scale_ratio),
    batch_size=opt.batch_size, num_workers=opt.n_cpu, shuffle=True, drop_last=True
)

val_dataloader = DataLoader(
    ImageDataset(base_dataset_dir=opt.dataset_dir, mode="test", normalize=opt.normalize, datatype=opt.datatype, seed=opt.seed, size=opt.crop_size, test_idx=None, augmentation=False, equalize=equalize, scale_ratio=opt.scale_ratio),
    batch_size=1, num_workers=1, shuffle=False, drop_last=True
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

criterion_GAN = torch.nn.MSELoss()

def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    loss = torch.mean(mu_2)
    return loss

# Adversarial ground truths
valid = 1
fake = 0

tot_i = 0
# Train loop
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

            # Adversarial ground truths
            if opt.discriminator_type == "patch":
                valid = Variable(Tensor(np.ones((X_1.size(0), *D1.output_shape))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((X_1.size(0), *D1.output_shape))), requires_grad=False)
            elif opt.discriminator_type == "multi":
                valid = 1
                fake = 0

            # Iniital feature maps
            if opt.lambda_kl_1 > 0 or opt.lambda_kl_2 > 0:
                mu_Z_1, Z_1 = Enc1(X_1)
                mu_Z_2, Z_2 = Enc2(X_2)
            else:
                Z_1 = Enc1(X_1)
                Z_2 = Enc2(X_2)
            # mu_Z_1, Z_1 = Enc1(X_1)
            # mu_Z_2, Z_2 = Enc2(X_2)

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
            # mu_Z_12, Z_12 = Enc2(X_12)
            # mu_Z_21, Z_21 = Enc1(X_21)

            # Cycle reconstruction
            X_121 = Dec1(Z_12)
            X_212 = Dec2(Z_21)


            # Compute Losses
            optimizer_G.zero_grad()

            # Total Loss
            loss_G = 0.0

            # Image Domain Adversarial Losses
            if opt.discriminator_type == "patch":
                loss_adv_1 = criterion_GAN(D1(X_21), valid)
                if opt.lambda_img_adv_1 > 0:
                    loss_G += opt.lambda_img_adv_1 * loss_adv_1
                loss_adv_2 = criterion_GAN(D2(X_12), valid)
                if opt.lambda_img_adv_2 > 0:
                    loss_G += opt.lambda_img_adv_2 * loss_adv_2
            elif opt.discriminator_type == "multi":
                loss_adv_1 = compute_loss_D_adv(D1, X_21, valid)
                if opt.lambda_img_adv_1 > 0:
                    loss_G += opt.lambda_img_adv_1 * loss_adv_1
                loss_adv_2 = compute_loss_D_adv(D2, X_12, valid)
                if opt.lambda_img_adv_2 > 0:
                    loss_G += opt.lambda_img_adv_2 * loss_adv_2
            
            # Common feature map reconstruction loss
            loss_com_rec_1 = criterion_recon(Z_12, Z_1.detach()) 
            if opt.lambda_com_rec_1 > 0:
                loss_G += opt.lambda_com_rec_1 * loss_com_rec_1
            loss_com_rec_2 = criterion_recon(Z_21, Z_2.detach())
            if opt.lambda_com_rec_2 > 0:
                loss_G += opt.lambda_com_rec_2 * loss_com_rec_2

            # Image reconstruction loss
            loss_img_rec_1 = criterion_recon(X_11, X_1)
            if opt.lambda_img_rec_1 > 0:
                loss_G += opt.lambda_img_rec_1 * loss_img_rec_1
            loss_img_rec_2 = criterion_recon(X_22, X_2)
            if opt.lambda_img_rec_2 > 0:
                loss_G += opt.lambda_img_rec_2 * loss_img_rec_2

            # Image cycle reconstruction loss
            loss_img_cyc_1 = criterion_recon(X_121, X_1)
            if opt.lambda_img_cyc_1 > 0:
                loss_G += opt.lambda_img_cyc_1 * loss_img_cyc_1
            loss_img_cyc_2 = criterion_recon(X_212, X_2)
            if opt.lambda_img_cyc_2 > 0:
                loss_G += opt.lambda_img_cyc_2 * loss_img_cyc_2
            
            if opt.lambda_kl_1 > 0:
                loss_kl_1 = compute_kl(mu_Z_1)
                loss_kl_2 = compute_kl(mu_Z_2)
                # KL Regularization
                loss_G += opt.lambda_kl_1 * loss_kl_1 + opt.lambda_kl_1 * loss_kl_2
            else:
                loss_kl_1 = 0
                loss_kl_2 = 0
            
            if opt.lambda_kl_2 > 0:
                loss_kl_12 = compute_kl(mu_Z_12)
                loss_kl_21 = compute_kl(mu_Z_21)
                # KL Regularization
                loss_G += opt.lambda_kl_2 * loss_kl_12 + opt.lambda_kl_2 * loss_kl_21
            else:
                loss_kl_12 = 0
                loss_kl_21 = 0

            loss_l1_1 = torch.mean(torch.abs(Z_1))
            loss_l1_2 = torch.mean(torch.abs(Z_2))
            if opt.lambda_l1_1 > 0:
                # L1 Regularization
                loss_G += opt.lambda_l1_1 * loss_l1_1 + opt.lambda_l1_1 * loss_l1_2

            loss_l1_12 = torch.mean(torch.abs(Z_12))
            loss_l1_21 = torch.mean(torch.abs(Z_21))
            if opt.lambda_l1_2 > 0:
                # L1 Regularization
                loss_G += opt.lambda_l1_2 * loss_l1_12 + opt.lambda_l1_2 * loss_l1_21

            # Randomly apply rotations and flips
            if opt.lambda_aug > 0:
                X_1_aug, X_1_rot = random_rotate(X_1)
                X_1_aug, X_1_flip = random_flip(X_1_aug)

                X_2_aug, X_2_rot = random_rotate(X_2)
                X_2_aug, X_2_flip = random_flip(X_2_aug)

                # Pass augmented images through the network
                if opt.lambda_kl_1 > 0 or opt.lambda_kl_2 > 0:
                    mu_Z_1_aug, Z_1_aug = Enc1(X_1_aug)
                    mu_Z_2_aug, Z_2_aug = Enc2(X_2_aug)
                else:
                    Z_1_aug = Enc1(X_1_aug)
                    Z_2_aug = Enc2(X_2_aug)
                
                # Translate images to opposite domain
                X_12_aug = Dec2(Z_1_aug)
                X_21_aug = Dec1(Z_2_aug)

                # Compute Similarity Loss with augmented translated images
                X_12_aug_gt = transforms.functional.rotate(X_12, X_1_rot)
                if X_1_flip[0]:
                    X_12_aug_gt = transforms.functional.hflip(X_12_aug_gt)
                if X_1_flip[1]:
                    X_12_aug_gt = transforms.functional.vflip(X_12_aug_gt)
                loss_aug_1 = opt.lambda_aug * criterion_recon(X_12_aug, X_12_aug_gt)

                X_21_aug_gt = transforms.functional.rotate(X_21, X_2_rot)
                if X_2_flip[0]:
                    X_21_aug_gt = transforms.functional.hflip(X_21_aug_gt)
                if X_2_flip[1]:
                    X_21_aug_gt = transforms.functional.vflip(X_21_aug_gt)
                loss_aug_2 = opt.lambda_aug * criterion_recon(X_21_aug, X_21_aug_gt)
                
                loss_G += loss_aug_1 + loss_aug_2
            else:
                loss_aug_1 = 0
                loss_aug_2 = 0

            # Total loss
            # loss_G = (
            #     loss_adv_1
            #     + loss_adv_2
            #     + loss_com_rec_1
            #     + loss_com_rec_2
            #     + loss_img_rec_1
            #     + loss_img_rec_2
            #     + loss_img_cyc_1
            #     + loss_img_cyc_2
            #     + loss_kl_1
            #     + loss_kl_2
            #     + loss_kl_12
            #     + loss_kl_21
            #     + loss_aug_1
            #     + loss_aug_2
            # )

            loss_G.backward()
            if opt.clip_mode == "value":
                torch.nn.utils.clip_grad_value_(Enc1.parameters(), clip_value=opt.clip_value)
                torch.nn.utils.clip_grad_value_(Dec1.parameters(), clip_value=opt.clip_value)
                torch.nn.utils.clip_grad_value_(Enc2.parameters(), clip_value=opt.clip_value)
                torch.nn.utils.clip_grad_value_(Dec2.parameters(), clip_value=opt.clip_value)
            elif opt.clip_mode == "norm":
                torch.nn.utils.clip_grad_norm_(Enc1.parameters(), opt.clip_value)
                torch.nn.utils.clip_grad_norm_(Dec1.parameters(), opt.clip_value)
                torch.nn.utils.clip_grad_norm_(Enc2.parameters(), opt.clip_value)
                torch.nn.utils.clip_grad_norm_(Dec2.parameters(), opt.clip_value)
            optimizer_G.step()

            if epoch > opt.D_delay:
                # Train Input Domain Discriminator
                optimizer_D1.zero_grad()

                if opt.discriminator_type == "patch":
                    loss_D1 = (
                        criterion_GAN(D1(X_1), valid)
                        + criterion_GAN(D1(X_21.detach()), fake)
                    )
                elif opt.discriminator_type == "multi":
                    loss_D1 = (
                        compute_loss_D_adv(D1, X_1, valid)
                        + compute_loss_D_adv(D1, X_21.detach(), fake)
                    )

                loss_D1.backward()
                if opt.clip_mode == "value":
                    torch.nn.utils.clip_grad_value_(D1.parameters(), clip_value=opt.clip_value)
                elif opt.clip_mode == "norm":
                    torch.nn.utils.clip_grad_norm_(D1.parameters(), opt.clip_value)
                optimizer_D1.step()

                # Train Output Domain Discriminator
                optimizer_D2.zero_grad()

                
                if opt.discriminator_type == "patch":
                    loss_D2 = (
                        criterion_GAN(D2(X_2), valid)
                        + criterion_GAN(D2(X_12.detach()), fake)
                    )
                elif opt.discriminator_type == "multi":
                    loss_D2 = (
                        compute_loss_D_adv(D2, X_2, valid)
                        + compute_loss_D_adv(D2, X_12.detach(), fake)
                    )

                loss_D2.backward()
                if opt.clip_mode == "value":
                    torch.nn.utils.clip_grad_value_(D2.parameters(), clip_value=opt.clip_value)
                elif opt.clip_mode == "norm":
                    torch.nn.utils.clip_grad_norm_(D2.parameters(), opt.clip_value)
                optimizer_D2.step()

            else:
                loss_D1 = 0
                loss_D2 = 0

        # # Log
        batches_done = epoch*opt.iter_per_epoch + e_i

        if (e_i + 1) % opt.log_loss_interval == 0:
            writer.add_scalar("00.Overall (Train)/01. Total Generator Loss", loss_G, batches_done)
            writer.add_scalar("00.Overall (Train)/02. Total Discriminator A->B Loss", loss_D1, batches_done)
            writer.add_scalar("00.Overall (Train)/03. Total Discriminator B->A Loss", loss_D2, batches_done)

            writer.add_scalar("01.Generator A->B (Train)/01. Image Domain Adversarial Loss A->B", loss_adv_2, batches_done)
            writer.add_scalar("01.Generator A->B (Train)/03. Common Feature Reconstruction Loss A->B", loss_com_rec_1, batches_done)
            writer.add_scalar("01.Generator A->B (Train)/05. Image Reconstruction Loss A->B", loss_img_rec_1, batches_done)
            writer.add_scalar("01.Generator A->B (Train)/06. Image Cycle Reconstruction Loss A->B", loss_img_cyc_1, batches_done)
            writer.add_scalar("01.Generator A->B (Train)/07. Augmentation Loss A->B", loss_aug_1, batches_done)

            writer.add_scalar("02.Generator B->A (Train)/01. Image Domain Adversarial Loss B->A", loss_adv_1, batches_done)
            writer.add_scalar("02.Generator B->A (Train)/03. Common Feature Reconstruction Loss B->A", loss_com_rec_2, batches_done)
            writer.add_scalar("02.Generator B->A (Train)/05. Image Reconstruction Loss B->A", loss_img_rec_2, batches_done)
            writer.add_scalar("02.Generator B->A (Train)/06. Image Cycle Reconstruction Loss B->A", loss_img_cyc_2, batches_done)
            writer.add_scalar("02.Generator B->A (Train)/07. Augmentation Loss B->A", loss_aug_2, batches_done)

            writer.add_scalar("01.Generator A->B (Train)/08. KL Regularization Loss A->B", loss_kl_1, batches_done)
            writer.add_scalar("02.Generator B->A (Train)/08. KL Regularization Loss B->A", loss_kl_2, batches_done)

            writer.add_scalar("01.Generator A->B (Train)/09. KL Regularization Loss A->B (Translated)", loss_kl_12, batches_done)
            writer.add_scalar("02.Generator B->A (Train)/09. KL Regularization Loss B->A (Translated)", loss_kl_21, batches_done)

            writer.add_scalar("01.Generator A->B (Train)/10. L1 Regularization Loss A->B", loss_l1_1, batches_done)
            writer.add_scalar("02.Generator B->A (Train)/10. L1 Regularization Loss B->A", loss_l1_2, batches_done)

            writer.add_scalar("01.Generator A->B (Train)/11. L1 Regularization Loss A->B (Translated)", loss_l1_12, batches_done)
            writer.add_scalar("02.Generator B->A (Train)/11. L1 Regularization Loss B->A (Translated)", loss_l1_21, batches_done)

        if (e_i + 1) % opt.log_image_interval == 0:
            X_1_overlay = overlay_images(X_12[0], X_1[0])
            X_1_overlay_grid = make_grid(X_1_overlay, normalize=True)

            X_2_overlay = overlay_images(X_21[0], X_2[0])
            X_2_overlay_grid = make_grid(X_2_overlay, normalize=True)

            X_1_grid = make_grid(X_1[0], normalize=True, value_range=(0, X_1[0].max().item()))
            X_2_grid = make_grid(X_2[0], normalize=True, value_range=(0, X_2[0].max().item()))

            X_11_grid = make_grid(X_11[0], normalize=True, value_range=(0, X_11[0].max().item()))
            X_22_grid = make_grid(X_22[0], normalize=True, value_range=(0, X_22[0].max().item()))
            
            X_12_grid = make_grid(X_12[0], normalize=True, value_range=(0, X_12[0].max().item()))
            X_21_grid = make_grid(X_21[0], normalize=True, value_range=(0, X_21[0].max().item()))

            X_121_grid = make_grid(X_121[0], normalize=True, value_range=(0, X_121[0].max().item()))
            X_212_grid = make_grid(X_212[0], normalize=True, value_range=(0, X_212[0].max().item()))

            writer.add_image("01. Cytosolic->NLS (Train)/00. Overlay Cytosolic Image (X_12)", X_1_overlay_grid, batches_done)
            writer.add_image("01. Cytosolic->NLS (Train)/01. Real Cytosolic Image (X_1)", X_1_grid, batches_done)
            for ch in range(Z_1.shape[1]):
                Z_1_common_grid = make_grid(Z_1[0,ch,:,:], normalize=True)
                writer.add_image(f"01. Cytosolic->NLS (Train) Z Channels 1 /02_{ch}. Common Feature CH {ch} (Z_1_common)", Z_1_common_grid, batches_done)
            writer.add_image("01. Cytosolic->NLS (Train)/03. Reconstructed Cytosolic Image (X_11)", X_11_grid, batches_done)
            writer.add_image("01. Cytosolic->NLS (Train)/04. Translated NLS Image (X_12)", X_12_grid, batches_done)
            for ch in range(Z_12.shape[1]):
                Z_12_common_grid = make_grid(Z_12[0,ch,:,:], normalize=True)
                writer.add_image(f"01. Cytosolic->NLS (Train) Z Channels 2 /05_{ch}. Translated Common Feature CH {ch} (Z_12_common)", Z_12_common_grid, batches_done)
            writer.add_image("01. Cytosolic->NLS (Train)/06. Cycle Reconstructed Cytosolic Image (X_121)", X_121_grid, batches_done)

            writer.add_image("02. NLS->Cytosolic (Train)/00. Overlay NLS Image (X_21)", X_2_overlay_grid, batches_done)
            writer.add_image("02. NLS->Cytosolic (Train)/01. Real NLS Image (X_2)", X_2_grid, batches_done)
            for ch in range(Z_2.shape[1]):
                Z_2_common_grid = make_grid(Z_2[0,ch,:,:], normalize=True)
                writer.add_image(f"02. NLS->Cytosolic Z Channels 1 (Train)/02_{ch}. Common Feature CH {ch} (Z_2_common)", Z_2_common_grid, batches_done)
            writer.add_image("02. NLS->Cytosolic (Train)/03. Reconstructed NLS Image (X_22)", X_22_grid, batches_done)
            writer.add_image("02. NLS->Cytosolic (Train)/04. Translated Cytosolic Image (X_21)", X_21_grid, batches_done)
            for ch in range(Z_21.shape[1]):
                Z_21_common_grid = make_grid(Z_21[0,ch,:,:], normalize=True)
                writer.add_image(f"02. NLS->Cytosolic Z Channels 2 (Train)/05_{ch}. Translated Common Feature CH {ch} (Z_21_common)", Z_21_common_grid, batches_done)
            writer.add_image("02. NLS->Cytosolic (Train)/06. Cycle Reconstructed NLS Image (X_212)", X_212_grid, batches_done)

            if opt.lambda_aug > 0:
                X_1_aug_grid = make_grid(X_1_aug[0], normalize=True, value_range=(0, X_1_aug[0].max().item()))
                X_12_aug_grid = make_grid(X_12_aug[0], normalize=True, value_range=(0, X_12_aug[0].max().item()))
                X_12_aug_gt_grid = make_grid(X_12_aug_gt[0], normalize=True, value_range=(0, X_12_aug_gt[0].max().item()))
                X_12_aug_overlay = overlay_images(X_12_aug[0], X_12_aug_gt[0])
                X_12_aug_overlay_grid = make_grid(X_12_aug_overlay, normalize=True)

                writer.add_image("01. Cytosolic->NLS Augmented (Train)/07. Augmented Cytosolic Image (X_1_aug)", X_1_aug_grid, batches_done)
                writer.add_image("01. Cytosolic->NLS Augmented (Train)/08. Augmented Translated NLS Image (X_12_aug)", X_12_aug_grid, batches_done)
                writer.add_image("01. Cytosolic->NLS Augmented (Train)/09. Augmented Translated NLS Image GT (X_12_aug_gt)", X_12_aug_gt_grid, batches_done)
                writer.add_image("01. Cytosolic->NLS Augmented (Train)/10. Augmented Overlay Cytosolic Image (X_12_aug)", X_12_aug_overlay_grid, batches_done)

                X_2_aug_grid = make_grid(X_2_aug[0], normalize=True, value_range=(0, X_2_aug[0].max().item()))
                X_21_aug_grid = make_grid(X_21_aug[0], normalize=True, value_range=(0, X_21_aug[0].max().item()))
                X_21_aug_gt_grid = make_grid(X_21_aug_gt[0], normalize=True, value_range=(0, X_21_aug_gt[0].max().item()))
                X_21_aug_overlay = overlay_images(X_21_aug[0], X_21_aug_gt[0])
                X_21_aug_overlay_grid = make_grid(X_21_aug_overlay, normalize=True)

                writer.add_image("02. NLS->Cytosolic Augmented (Train)/07. Augmented NLS Image (X_2_aug)", X_2_aug_grid, batches_done)
                writer.add_image("02. NLS->Cytosolic Augmented (Train)/08. Augmented Translated Cytosolic Image (X_21_aug)", X_21_aug_grid, batches_done)
                writer.add_image("02. NLS->Cytosolic Augmented (Train)/09. Augmented Translated Cytosolic Image GT (X_21_aug_gt)", X_21_aug_gt_grid, batches_done)
                writer.add_image("02. NLS->Cytosolic Augmented (Train)/10. Augmented Overlay NLS Image (X_21_aug)", X_21_aug_overlay_grid, batches_done)
    
    # Update learning rates
    # lr_scheduler_G.step()
    # lr_scheduler_D1.step()
    # lr_scheduler_D2.step()

    Enc1.eval()
    Dec1.eval()
    Enc2.eval()
    Dec2.eval()

    if epoch % opt.val_interval == 0:
        with torch.no_grad():
            batch = next(iter(val_dataloader))

            X_1 = Variable(batch["A"].type(Tensor)).to(device)
            X_2 = Variable(batch["B"].type(Tensor)).to(device)

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

            X_1_overlay = overlay_images(X_12[0], X_1[0])
            X_1_overlay_grid = make_grid(X_1_overlay, normalize=True)

            X_2_overlay = overlay_images(X_21[0], X_2[0])
            X_2_overlay_grid = make_grid(X_2_overlay, normalize=True)

            X_1_grid = make_grid(X_1, normalize=True, value_range=(0, X_1.max().item()))
            X_2_grid = make_grid(X_2, normalize=True, value_range=(0, X_2.max().item()))

            X_11_grid = make_grid(X_11, normalize=True, value_range=(0, X_11.max().item()))
            X_22_grid = make_grid(X_22, normalize=True, value_range=(0, X_22.max().item()))
            
            X_12_grid = make_grid(X_12, normalize=True, value_range=(0, X_12.max().item()))
            X_21_grid = make_grid(X_21, normalize=True, value_range=(0, X_21.max().item()))

            X_121_grid = make_grid(X_121, normalize=True, value_range=(0, X_121.max().item()))
            X_212_grid = make_grid(X_212, normalize=True, value_range=(0, X_212.max().item()))

            writer.add_image("01. Cytosolic->NLS (Val)/00. Overlay Cytosolic Image (X_12)", X_1_overlay_grid, epoch)
            writer.add_image("01. Cytosolic->NLS (Val)/01. Real Cytosolic Image (X_1)", X_1_grid, epoch)
            for ch in range(Z_1.shape[1]):
                Z_1_common_grid = make_grid(Z_1[0,ch,:,:], normalize=True)
                writer.add_image(f"01. Cytosolic->NLS (Val)/02_{ch}. Common Feature CH {ch} (Z_1_common)", Z_1_common_grid, epoch)
            writer.add_image("01. Cytosolic->NLS (Val)/03. Reconstructed Cytosolic Image (X_11)", X_11_grid, epoch)
            writer.add_image("01. Cytosolic->NLS (Val)/04. Translated NLS Image (X_12)", X_12_grid, epoch)
            for ch in range(Z_12.shape[1]):
                Z_12_common_grid = make_grid(Z_12[0,ch,:,:], normalize=True)
                writer.add_image(f"01. Cytosolic->NLS (Val)/05_{ch}. Translated Common Feature CH {ch} (Z_12_common)", Z_12_common_grid, epoch)
            writer.add_image("01. Cytosolic->NLS (Val)/06. Cycle Reconstructed Cytosolic Image (X_121)", X_121_grid, epoch)

            writer.add_image("02. NLS->Cytosolic (Val)/00. Overlay NLS Image (X_21)", X_2_overlay_grid, epoch)
            writer.add_image("02. NLS->Cytosolic (Val)/01. Real NLS Image (X_2)", X_2_grid, epoch)
            for ch in range(Z_2.shape[1]):
                Z_2_common_grid = make_grid(Z_2[0,ch,:,:], normalize=True)
                writer.add_image(f"02. NLS->Cytosolic (Val)/02_{ch}. Common Feature CH {ch} (Z_2_common)", Z_2_common_grid, epoch)
            writer.add_image("02. NLS->Cytosolic (Val)/03. Reconstructed NLS Image (X_22)", X_22_grid, epoch)
            writer.add_image("02. NLS->Cytosolic (Val)/04. Translated Cytosolic Image (X_21)", X_21_grid, epoch)
            for ch in range(Z_21.shape[1]):
                Z_21_common_grid = make_grid(Z_21[0,ch,:,:], normalize=True)
                writer.add_image(f"02. NLS->Cytosolic (Val)/05_{ch}. Translated Common Feature CH {ch} (Z_21_common)", Z_21_common_grid, epoch)
            writer.add_image("02. NLS->Cytosolic (Val)/06. Cycle Reconstructed NLS Image (X_212)", X_212_grid, epoch)
    
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
        # torch.save(lr_scheduler_G.state_dict(), f"{saved_models_dir}/lr_scheduler_G_{epoch}.pth")
        # torch.save(lr_scheduler_D1.state_dict(), f"{saved_models_dir}/lr_scheduler_D1_{epoch}.pth")
        # torch.save(lr_scheduler_D2.state_dict(), f"{saved_models_dir}/lr_scheduler_D2_{epoch}.pth")
