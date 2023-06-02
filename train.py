"""

TODO List:
1. Tensorboard

"""

import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from skimage.color import rgb2gray
from torch.utils.tensorboard import SummaryWriter

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from pytorch_msssim import ssim, ms_ssim

from logger import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=301, help="number of epochs of training")

parser.add_argument("--dataset_name", type=str, default="Cytosolic2NLS_dataset_512_1x_1000_1000", help="name of the dataset")
parser.add_argument("--experiment_name", type=str, default="testMultiD", help="name of the experiment")

parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--step_size", type=int, default=100, help="adam: decay of first order momentum of gradient")
parser.add_argument("--gamma", type=float, default=0.5, help="how much to decay learning rate")
parser.add_argument("--decay_epoch", type=int, default=200, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--num_scales", type=int, default=3, help="number of scales for the multi-scale discriminator")
parser.add_argument("--downsample_stride", type=int, default=3, help="stride for downsample for the multi-scale discriminator (ex: 2=1/2, 4=1/4)")
parser.add_argument("--sample_interval", type=int, default=250, help="interval saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=25, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--log_dir", type=str, default='./log', help="path to save log")
parser.add_argument("--unaligned", type=bool, default=True, help="setting to align image pairs")

parser.add_argument("--lambda_adv_img", type=int, default=5, help="Hyperparameter for adversarial loss at image level")
parser.add_argument("--lambda_adv_feat", type=int, default=5, help="Hyperparameter for adversarial loss at feature level")
parser.add_argument("--lambda_img_rec", type=int, default=100, help="Hyperparameter for reconstruction loss at image level")
parser.add_argument("--lambda_feat_rec", type=int, default=100, help="Hyperparameter for reconstruction loss at feature level")
parser.add_argument("--lambda_cyc", type=int, default=100, help="Hyperparameter for cycle consistency loss")
parser.add_argument("--lambda_id", type=int, default=1, help="Hyperparameter for identity loss")

parser.add_argument("--in_channels", type=int, default=1, help="number of channels of input images")
parser.add_argument("--out_channels", type=int, default=1, help="number of channels of output images")
parser.add_argument("--feat_channels", type=int, default=1, help="number of channels of feature")

parser.add_argument('--in_datatype', type=str, default='f32', help='datatype [f32 | uint8]')
parser.add_argument('--out_datatype', type=str, default='f32', help='datatype [f32 | uint8]')

parser.add_argument('--tensorboard_dir', type=str, default='./runs', help='Log directory for Tensorboard')

parser.add_argument("--discriminator_img_rate", type=int, default=1, help="Update rate of image discriminator training")
parser.add_argument("--discriminator_feat_rate", type=int, default=1, help="Update rate of feature discriminator training")
parser.add_argument("--generator_rate", type=int, default=1, help="Update rate of generator training")

parser.add_argument("--D_Z_loss", type=str, default='CE', help="Loss for feature discriminator [BCE | CE]")
parser.add_argument("--feat_rec_loss", type=str, default='correlation', help="Loss for feature reconstruction [L1 | correlation | SSIM]")
parser.add_argument("--D_Z_last_layer", type=str, default='adaptAvgPool', help="Last layer for feature discriminator [adaptAvgPool | linear]")

parser.add_argument("--arch", type=str, default='AE', help="Architecture for Encoder/Decoder [AE | UNET]")
parser.add_argument("--unet_n_downsample", type=int, default=8, help="Number of downsampling for UNET") # 9 for 1x1 in bottleneck for 512 input

parser.add_argument("--invert", type=bool, default=False, help="Invert input")


parser.add_argument("--Enc_activation", type=str, default='relu', help="Output activation function for for Encoder/Decoder [tanh | relu]")
parser.add_argument("--Dec_activation", type=str, default='tanh', help="Output activation function for for Encoder/Decoder [tanh | relu]")

parser.add_argument("--save_images", type=bool, default=False, help="enable saving test sampless")

opt = parser.parse_args()
print(opt)

# Writer for TensorBoard
writer = SummaryWriter(log_dir=opt.tensorboard_dir)

writer.add_text('Options', str(opt), 0)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.experiment_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.experiment_name, exist_ok=True)

# Initialize encoders, generators and discriminators
if opt.arch == "AE":
    Enc1 = Encoder(in_channels=opt.in_channels, feat_channels=opt.feat_channels, dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, output_activation=opt.Enc_activation)
    Dec1 = Decoder(feat_channels=opt.feat_channels, out_channels=opt.in_channels, n_residual=opt.n_residual, output_activation=opt.Dec_activation)
    Enc2 = Encoder(in_channels=opt.out_channels, feat_channels=opt.feat_channels, dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, output_activation=opt.Enc_activation)
    Dec2 = Decoder(feat_channels=opt.feat_channels, out_channels=opt.out_channels, n_residual=opt.n_residual, output_activation=opt.Dec_activation)
elif opt.arch == 'UNET':
    Enc1 = UnetGenerator(input_nc=opt.in_channels, output_nc=opt.feat_channels, num_downs=opt.unet_n_downsample, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False)
    Dec1 = UnetGenerator(input_nc=opt.feat_channels, output_nc=opt.in_channels, num_downs=opt.unet_n_downsample, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
    Enc2 = UnetGenerator(input_nc=opt.out_channels, output_nc=opt.feat_channels, num_downs=opt.unet_n_downsample, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
    Dec2 = UnetGenerator(input_nc=opt.feat_channels, output_nc=opt.out_channels, num_downs=opt.unet_n_downsample, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)

D1 = MultiDiscriminator(channels=opt.in_channels, num_scales=opt.num_scales, downsample_stride=opt.downsample_stride)
D2 = MultiDiscriminator(channels=opt.out_channels, num_scales=opt.num_scales, downsample_stride=opt.downsample_stride)
D_Z = MultiScaleMultiClassDiscriminator(channels=opt.feat_channels, num_classes=4, num_scales=opt.num_scales, downsample_stride=opt.downsample_stride, last_layer=opt.D_Z_last_layer)

# Upload models to GPU
cuda = torch.cuda.is_available()
if cuda:
    Enc1 = nn.DataParallel(Enc1)
    Enc1.cuda()
    Dec1 = nn.DataParallel(Dec1)
    Dec1.cuda()
    Enc2 = nn.DataParallel(Enc2)
    Enc2.cuda()
    Dec2 = nn.DataParallel(Dec2)
    Dec2.cuda()
    D1 = nn.DataParallel(D1)
    D1.cuda()
    D2 = nn.DataParallel(D2)
    D2.cuda()
    D_Z = nn.DataParallel(D_Z)
    D_Z.cuda()
    # Enc1 = nn.DataParallel(Enc1, device_ids = [0, 1, 2])
    # Enc1.cuda()
    # Dec1 = nn.DataParallel(Dec1, device_ids = [0, 1, 2])
    # Dec1.cuda()
    # Enc2 = nn.DataParallel(Enc2, device_ids = [0, 1, 2])
    # Enc2.cuda()
    # Dec2 = nn.DataParallel(Dec2, device_ids = [0, 1, 2])
    # Dec2.cuda()
    # D1 = nn.DataParallel(D1, device_ids = [0, 1, 2])
    # D1.cuda()
    # D2 = nn.DataParallel(D2, device_ids = [0, 1, 2])
    # D2.cuda()
    # D_Z = nn.DataParallel(D_Z, device_ids = [0, 1, 2])
    # D_Z.cuda()


    # D1 = D1.cuda()
    # D2 = D2.cuda()
    # D_Z = D_Z.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


# Set up models
if opt.epoch != 0:
    # Load pretrained models
    Enc1.load_state_dict(torch.load("saved_models/%s/Enc1_%d.pth" % (opt.experiment_name, opt.epoch)))
    Dec1.load_state_dict(torch.load("saved_models/%s/Dec1_%d.pth" % (opt.experiment_name, opt.epoch)))
    Enc2.load_state_dict(torch.load("saved_models/%s/Enc2_%d.pth" % (opt.experiment_name, opt.epoch)))
    Dec2.load_state_dict(torch.load("saved_models/%s/Dec2_%d.pth" % (opt.experiment_name, opt.epoch)))
    D1.load_state_dict(torch.load("saved_models/%s/D1_%d.pth" % (opt.experiment_name, opt.epoch)))
    D2.load_state_dict(torch.load("saved_models/%s/D2_%d.pth" % (opt.experiment_name, opt.epoch)))
    D_Z.load_state_dict(torch.load("saved_models/%s/D2_%d.pth" % (opt.experiment_name, opt.epoch)))
else:
    # Initialize weights
    Enc1.apply(weights_init('kaiming'))
    Dec1.apply(weights_init('kaiming'))
    Enc2.apply(weights_init('kaiming'))
    Dec2.apply(weights_init('kaiming'))
    D1.apply(weights_init('gaussian'))
    D2.apply(weights_init('gaussian'))
    D_Z.apply(weights_init('gaussian'))

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters()),
    lr=opt.lr,
    betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay
)
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
optimizer_D_Z = torch.optim.Adam(D_Z.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=opt.step_size,
                                        gamma=opt.gamma, last_epoch=-1)
lr_scheduler_D1 = torch.optim.lr_scheduler.StepLR(optimizer_D1, step_size=opt.step_size,
                                        gamma=opt.gamma, last_epoch=-1)
lr_scheduler_D2 = torch.optim.lr_scheduler.StepLR(optimizer_D2, step_size=opt.step_size,
                                        gamma=opt.gamma, last_epoch=-1)
lr_scheduler_D_Z = torch.optim.lr_scheduler.StepLR(optimizer_D_Z, step_size=opt.step_size,
                                        gamma=opt.gamma, last_epoch=-1)

# Generate a random input tensor X1_dum SIZ_dumE: torch.Siz_dume([4, 1, 512, 512]) [batch_cnt, ch, x, y]
# X1_dum = torch.randn(opt.batch_size, opt.in_channels, 512, 512)
# X2_dum = torch.randn(opt.batch_size, opt.out_channels, 512, 512)

# Z_dum = torch.randn(opt.batch_size, opt.feat_channels, 512, 512)

# # Write the generator graph
# writer.add_graph(Enc1, X1_dum)
# writer.add_graph(Enc2, X2_dum)

# writer.add_graph(Dec1, Z_dum)
# writer.add_graph(Dec2, Z_dum)

# # Learning rate update schedulers
# optimizer_G = torch.optim.Adam(
#     itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters()),
#     lr=opt.lr, betas=(opt.b1, opt.b2)
# )
# optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_D_Z = torch.optim.Adam(D_Z.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
#     optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
# )
# lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
#     optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
# )
# lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
#     optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
# )
# lr_scheduler_D_Z = torch.optim.lr_scheduler.LambdaLR(
#     optimizer_D_Z, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
# )

# Configure dataloaders
dataloader = DataLoader(
    ImageDataset("./dataset/%s" % opt.dataset_name, mode="train", unaligned=opt.unaligned,in_datatype=opt.in_datatype,out_datatype=opt.out_datatype),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    ImageDataset("./dataset/%s" % opt.dataset_name, mode="test", unaligned=opt.unaligned,in_datatype=opt.in_datatype,out_datatype=opt.out_datatype),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=1,
)

def sample_images(batches_done, global_it):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    # Create copies of image
    X1 = Variable(imgs["A"].type(Tensor))
    X2 = Variable(imgs["B"].type(Tensor))

    # print(X1.size())
    # print(X1[0].unsqueeze(0).size())
    X1 = X1[0].unsqueeze(0)
    X2 = X2[0].unsqueeze(0)

    if opt.invert:
        X1_in = -X1.clone()
        X2_in = -X2.clone()
    else:
        X1_in = X1.clone()
        X2_in = X2.clone()

    # 1. Pass input images [X1, X2] through the encoders to obtain the feature map
    Z1 = Enc1(X1_in)
    Z2 = Enc2(X2_in)

    # 2. Pass the feature maps [Z1, Z2] through the decoders to obtain the 
    #    translated images [X1_trans, X2_trans] and reconstructed images [X1_recon, X2_recon]
    X1_recon = Dec1(Z1)
    X2_trans = Dec2(Z1)

    X1_trans = Dec1(Z2)
    X2_recon = Dec2(Z2)

    # 3. Pass the translated images [X1_trans, X2_trans] through the encoder to obtain the
    #    translated image feature maps [Z1_trans, Z2_trans]
    Z2_trans = Enc2(X2_trans)
    Z1_trans = Enc1(X1_trans)


    # 4. Pass the translated image feature maps through the decoders to obtain the 
    #    reconstructed translated images [X1_trans_recon, X2_trans_recon]
    X1_trans_recon = Dec1(Z2_trans)
    X2_trans_recon = Dec2(Z1_trans)
    
    # Concatenate samples horisontally
    # Arange images along x-axis rgb2gray(A_img_rgb)
    # X1 = make_grid(X1, nrow=5, normalize=True)
    X1 = make_grid(X1, nrow=5, normalize=True)
    X2 = make_grid(X2, nrow=5, normalize=True)

    # Z1 = make_grid(Z1, nrow=5, normalize=True)
    # Z2 = make_grid(Z2, nrow=5, normalize=True)
    
    X1_recon = make_grid(X1_recon, nrow=5, normalize=True)
    X2_trans = make_grid(X2_trans, nrow=5, normalize=True)
    
    X1_trans = make_grid(X1_trans, nrow=5, normalize=True)
    X2_recon = make_grid(X2_recon, nrow=5, normalize=True)

    # Z2_trans = make_grid(Z2_trans, nrow=5, normalize=True)
    # Z1_trans = make_grid(Z1_trans, nrow=5, normalize=True)
    
    # print(X1_trans_recon.shape)
    X1_trans_recon = make_grid(X1_trans_recon, nrow=5, normalize=True)
    X2_trans_recon = make_grid(X2_trans_recon, nrow=5, normalize=True)

    writer.add_image("A2B/X1", X1, global_it)
    writer.add_image("B2A/X2", X2, global_it)

    writer.add_image("A2B/X2_trans", X2_trans, global_it)
    writer.add_image("B2A/X1_trans", X1_trans, global_it)

    writer.add_image("A2B/X1_recon", X1_recon, global_it)
    writer.add_image("B2A/X2_recon", X2_recon, global_it)

    writer.add_image("A2B/X1_trans_recon", X1_trans_recon, global_it)
    writer.add_image("B2A/X2_trans_recon", X2_trans_recon, global_it)

    # print(Z1.shape)
    # 4, 16, 512, 512


    #TODO: Add feature map visualization to tensorboard
    for ch in range(Z1.shape[1]):
        Z1_ch = make_grid(Z1[:,ch,:,:][:,None,:,:], nrow=5, normalize=True)
        # print(Z1_ch.shape) 4, 512, 512
        writer.add_image(f"Z1/Z1_{ch}", Z1_ch, global_it)
        Z2_trans_ch = make_grid(Z2_trans[:,ch,:,:][:,None,:,:], nrow=5, normalize=True)
        writer.add_image(f"Z2_trans/Z2_trans_{ch}", Z2_trans_ch, global_it)

        Z2_ch = make_grid(Z2[:,ch,:,:][:,None,:,:], nrow=5, normalize=True)
        writer.add_image(f"Z2/Z2_{ch}", Z2_ch, global_it)
        Z1_trans_ch = make_grid(Z1_trans[:,ch,:,:][:,None,:,:], nrow=5, normalize=True)
        writer.add_image(f"Z1_trans/Z1_trans_{ch}", Z1_trans_ch, global_it)

    # writer.add_graph(Enc1, images)
    # writer.add_graph(Enc2, images)
    # writer.add_graph(Dec1, images)
    # writer.add_graph(Dec2, images)
    # writer.add_graph(D1, images)
    # writer.add_graph(D2, images)
    # writer.add_graph(D_Z, images)


    # writer.add_graph(model, X1)

    # Concatenate with previous samples vertically
    if opt.save_images:
        image_grid = torch.cat((X1, Z1_ch, X1_recon, X2_trans, Z2_trans_ch, X1_trans_recon, X2, Z2_ch, X2_recon, X1_trans, Z1_trans_ch, X2_trans_recon), 1)
        save_image(image_grid, "images/%s/%s.png" % (opt.experiment_name, batches_done), nrow=5, normalize=False)



"""
Training
"""

# Adversarial ground truths
valid = 1
fake = 0

# Labels for feature discriminator
label_Z1_CE = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
label_Z2_CE = torch.tensor([0, 0, 1, 0], dtype=torch.float32)
label_Z1_trans_CE = torch.tensor([0, 1, 0, 0], dtype=torch.float32)
label_Z2_trans_CE = torch.tensor([1, 0, 0, 0], dtype=torch.float32)

label_fake_CE = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)

label_Z1_BCE = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)
label_Z2_BCE = torch.tensor([[0, 0, 1, 0]], dtype=torch.float32)
label_Z1_trans_BCE = torch.tensor([[0, 1, 0, 0]], dtype=torch.float32)
label_Z2_trans_BCE = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)

label_Z1_fake_BCE = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float32)
label_Z2_fake_BCE = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32)
label_Z1_trans_fake_BCE = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
label_Z2_trans_fake_BCE = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)

# if cuda:
    # label_Z1.cuda()
    # label_Z2.cuda()
    # label_Z1_trans.cuda()
    # label_Z2_trans.cuda()
    # label_fake.cuda()

# criterion_adv = torch.nn.MSELoss()
criterion_recon = torch.nn.L1Loss()
# criterion_D_Z = torch.nn.NLLLoss()

def compute_loss_D_adv(model, x, gt):
        """Computes the MSE between model output and scalar gt"""
        # loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        loss = 0
        n=0
        output = model.forward(x)
        for out in output:
            squared_diff = (out - gt) ** 2
            loss += torch.mean(squared_diff)
            # print(f"{n}: Mean squared_diff: {torch.mean(squared_diff)} and current loss: {loss}")
            # print(f"{n}: Out shape: {out.shape} ")
            n=n+1
        return loss




# def correlation_loss(y_pred, y_true):
#     x = y_pred.clone()
#     y = y_true.clone()
#     vx = x - torch.mean(x)
#     vy = y - torch.mean(y)
#     cov = torch.sum(vx * vy)
#     corr = cov / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
#     corr = torch.maximum(torch.minimum(corr,torch.tensor(1)), torch.tensor(-1))
#     return torch.sub(torch.tensor(1), corr ** 2)

def correlation_loss(y_pred, y_true):
    vx = y_pred - y_pred.mean()
    vy = y_true - y_true.mean()

    cov = torch.sum(vx * vy)

    corr = cov / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-12)
    corr = torch.clamp(corr, -1.0, 1.0)

    # Computing the loss with subtraction in-place
    loss = corr.pow_(2).neg_().add_(1.0)

    return loss
    
def SSIM_loss(y_pred, y_true):
    ms_ssim_loss = 1 - ms_ssim( y_pred, y_true, data_range=1, size_average=True )
    return ms_ssim_loss

criterion_D_Z_BCE = nn.BCEWithLogitsLoss()
def compute_loss_D_Z_BCE(model, input, label_list):
    # Assume model is your discriminator
    # input is the generator output
    # labels are the true labels of the generator

    # # Define the labels for the other generators, assuming there are four generators in total
    # label_list = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    # label_list.remove(label.tolist())  # Remove the true labels
    
    total_loss = 0
    cnt = 0
    output = model.forward(input)
    # print(f"Label list: {label_list}")
    for label in label_list:
        if cuda:
            label = label.cuda()
        for p_label in output:
            # print(cnt)
            # print(p_label.squeeze().size())
            # print(label.repeat(p_label.squeeze().size(0),1).size())
            total_loss += criterion_D_Z_BCE(p_label.squeeze(), label.repeat(p_label.squeeze().size(0),1))
            cnt += 1
            # print(f"output={p_label.squeeze()}, output shape={p_label.squeeze().size()}, output device={p_label.squeeze().device}")
            # print(f"label={label}, label shape={label.size()}, label device={p_label.device}")
            # print(f"WHAT{label.repeat(4,1).dtype}")
            # print(p_label.squeeze().size())
            # print(p_label.squeeze().size(0))

    return total_loss / cnt

criterion_D_Z_CE = torch.nn.CrossEntropyLoss()
def compute_loss_D_Z_CE(model, input, label):
    # input = input.type(torch.LongTensor)
    # label = label.type(torch.LongTensor)
    # print(f"input={input}")
    # print(f"label={label}")
    
    if cuda:
        # input = input.cuda()
        label = label.cuda()
    # input = input
    # label = label.long().cuda()

    # Pass the input tensor through the discriminator
    output = model.forward(input)
    # print(f"model device: {next(model.parameters()).device}")
    

    # Calculate the total loss by summing the losses from each scale
    total_loss = 0.0
    for p_label in output:
        # print(f"output={p_label.squeeze()}, output shape={p_label.squeeze().size()}, output device={p_label.squeeze().device}")
        # print(f"label={label}, label shape={label.size()}, label device={p_label.device}")
        # print(f"WHAT{label.repeat(4,1).dtype}")
        # print(p_label.squeeze().size())
        # print(p_label.squeeze().size(0))
        loss = criterion_D_Z_CE(p_label.squeeze(), label.repeat(p_label.squeeze().size(0),1))
        total_loss += loss

    return total_loss

# criterion_D_Z_BCE = torch.nn.BCEWithLogitsLoss()
# def compute_loss_D_Z(model, input, label):
#     # input = input.type(torch.LongTensor)
#     # label = label.type(torch.LongTensor)
#     # print(f"input={input}")
#     # print(f"label={label}")
    
#     if cuda:
#         # input = input.cuda()
#         label = label.cuda()
#     # input = input
#     # label = label.long().cuda()

#     # Pass the input tensor through the discriminator
#     output = model.forward(input)
#     # print(f"model device: {next(model.parameters()).device}")
    

#     # Calculate the total loss by summing the losses from each scale
#     total_loss = 0.0
#     for p_label in output:
#         # print(f"output={p_label.squeeze()}, output shape={p_label.squeeze().size()}, output device={p_label.squeeze().device}")
#         # print(f"label={label}, label shape={label.size()}, label device={p_label.device}")
#         # print(f"WHAT{label.repeat(4,1).dtype}")
#         # print(label.repeat(p_label.squeeze().size(0),1).size())
#         # print(label.repeat(p_label.squeeze().size(0),1))
#         # print(p_label.squeeze().size())
#         # print(p_label.squeeze())
#         loss = criterion_D_Z_BCE(p_label.squeeze(), label.repeat(p_label.squeeze().size(0),1))
#         total_loss += loss

#     return total_loss




# Loss weights hyperparameters
lambda_adv_img = opt.lambda_adv_img
lambda_adv_feat = opt.lambda_adv_feat
lambda_img_rec = opt.lambda_img_rec
lambda_feat_rec = opt.lambda_feat_rec
lambda_cyc = opt.lambda_cyc
lambda_id = opt.lambda_id

logger = setup_logger("ImageTranslation", os.path.join(opt.log_dir, opt.experiment_name), 0, filename='i2i_log.txt')
logger.info(opt)

global_i = 0

discriminator_img_update_counter = 0
discriminator_feat_update_counter = 0
generator_update_counter = 0

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    epoch_start_time = time.time()
    for i, batch in enumerate(dataloader):
        
        # Set model input
        X1 = Variable(batch["A"].type(Tensor))
        X2 = Variable(batch["B"].type(Tensor))

        if opt.invert:
            X1_in = -X1.clone()
            X2_in = -X2.clone()
        else:
            X1_in = X1.clone()
            X2_in = X2.clone()

        # print(f"X1 SIZE: {X1.shape}")

        """
        
        Train Encoders and Generators
        
        Forward:
        1. Pass input images [X1, X2] through the encoders to obtain the feature map [Z1, Z2]
        2. Pass the feature maps [Z1, Z2] through the decoders to obtain the 
            translated images [X1_trans, X2_trans] and reconstructed images [X1_recon, X2_recon]
        3. Pass the translated images [X1_trans, X2_trans] through the encoder to obtain the
            translated image feature maps [Z1_trans, Z2_trans]
        4. Pass the translated image feature maps through the decoders to obtain the 
            reconstructed translated images [X1_trans_recon, X2_trans_recon]
        
        Loss:
        1. Cycle Consistency Loss (x2: one for each domain): 
            a. criterion_recon(X1_trans_recon, X1)
            b. criterion_recon(X2_trans_recon, X2)
        2. Image Reconstruction Loss (x2: one for each domain): 
            a. criterion_recon(X1_recon, X1)
            b. criterion_recon(X2_recon, X2)
        3. Latent Feature Reconstruction Loss (x2: one for each domain): 
            a. criterion_recon(Z2_trans, Z1)
            b. criterion_recon(Z1_trans, Z2)
        4. Image Domain Discriminator Loss (x2: one for each domain): 
            a. D1.compute_loss(X1_trans, valid)
            b. D2.compute_loss(X2_trans, valid)
        5. Cross-domain Feature Consistency Discriminator Loss: 
            TODO: what is fake and what is valid, i.e. what is the groundtruth? 
            --> I guess since we just want the images to be indistinguishable, just use one as groundtruth?
            --> Define Z1 : valid, the rest fake

            D_Z.compute_loss(Z1, valid)
            D_Z.compute_loss(Z2, fake)
            D_Z.compute_loss(Z1_trans, fake)
            D_Z.compute_loss(Z2_trans, fake)

        """
        # Train Generator
        if generator_update_counter % opt.generator_rate == 0:
            # print("Generator train")
            optimizer_G.zero_grad()

            # Forward Pass

            # 1. Pass input images [X1, X2] through the encoders to obtain the feature map
            Z1 = Enc1(X1_in)
            Z2 = Enc2(X2_in)

            # 2. Pass the feature maps [Z1, Z2] through the decoders to obtain the 
            #    translated images [X1_trans, X2_trans] and reconstructed images [X1_recon, X2_recon]
            X1_recon = Dec1(Z1)
            X2_trans = Dec2(Z1)

            X1_trans = Dec1(Z2)
            X2_recon = Dec2(Z2)

            # 3. Pass the translated images [X1_trans, X2_trans] through the encoder to obtain the
            #    translated image feature maps [Z1_trans, Z2_trans]

            Z2_trans = Enc2(X2_trans)
            Z1_trans = Enc1(X1_trans)


            # 4. Pass the translated image feature maps through the decoders to obtain the 
            #    reconstructed translated images [X1_trans_recon, X2_trans_recon]
            X1_trans_recon = Dec1(Z2_trans)  if lambda_cyc > 0 else 0
            X2_trans_recon = Dec2(Z1_trans)  if lambda_cyc > 0 else 0

            
            # Loss Calculations

            # 1. Cycle Consistency Loss (x2: one for each domain): 
            loss_cycle_1 = lambda_cyc * criterion_recon(X1_trans_recon, X1)  if lambda_cyc > 0 else 0
            loss_cycle_2 = lambda_cyc * criterion_recon(X2_trans_recon, X2)  if lambda_cyc > 0 else 0

            # 2. Image Reconstruction Loss (x2: one for each domain): 
            loss_img_recon_1 = lambda_img_rec * criterion_recon(X1_recon, X1)
            loss_img_recon_2 = lambda_img_rec * criterion_recon(X2_recon, X2)

            # 3. Latent Feature Reconstruction Loss (x2: one for each domain): 
            if opt.feat_rec_loss == "L1":
                loss_feat_recon_1 = lambda_feat_rec * criterion_recon(Z2_trans, Z1)
                loss_feat_recon_2 = lambda_feat_rec * criterion_recon(Z1_trans, Z2)
            elif opt.feat_rec_loss == "correlation":
                loss_feat_recon_1 = lambda_feat_rec * correlation_loss(Z2_trans, Z1)
                loss_feat_recon_2 = lambda_feat_rec * correlation_loss(Z1_trans, Z2)
            elif opt.feat_rec_loss == "SSIM":
                loss_feat_recon_1 = lambda_feat_rec * SSIM_loss(Z2_trans, Z1)
                loss_feat_recon_2 = lambda_feat_rec * SSIM_loss(Z1_trans, Z2)

            #4. Image Domain Discriminator Loss (x2: one for each domain): 
            # loss_adv_1 = lambda_adv_img * D1.compute_loss(X1_trans, valid)
            # loss_adv_2 = lambda_adv_img * D2.compute_loss(X2_trans, valid)
            loss_adv_1 = lambda_adv_img * compute_loss_D_adv(D1, X1_trans, valid)
            loss_adv_2 = lambda_adv_img * compute_loss_D_adv(D2, X2_trans, valid)

            #5. Cross-domain Feature Consistency Discriminator Loss (Fake label : Make discriminator random guess):
            if opt.D_Z_loss == 'BCE':
                loss_adv_Z_1 = lambda_adv_feat * compute_loss_D_Z_BCE(D_Z, Z1, label_Z1_fake_BCE)
                loss_adv_Z_2 = lambda_adv_feat * compute_loss_D_Z_BCE(D_Z, Z2, label_Z2_fake_BCE)
                loss_adv_Z_trans_1 = lambda_adv_feat * compute_loss_D_Z_BCE(D_Z, Z1_trans, label_Z1_trans_fake_BCE)
                loss_adv_Z_trans_2 = lambda_adv_feat * compute_loss_D_Z_BCE(D_Z, Z2_trans, label_Z2_trans_fake_BCE)
            else:
                loss_adv_Z_1 = lambda_adv_feat * compute_loss_D_Z_CE(D_Z, Z1, label_fake_CE)
                loss_adv_Z_2 = lambda_adv_feat * compute_loss_D_Z_CE(D_Z, Z2, label_fake_CE)
                loss_adv_Z_trans_1 = lambda_adv_feat * compute_loss_D_Z_CE(D_Z, Z1_trans, label_fake_CE)
                loss_adv_Z_trans_2 = lambda_adv_feat * compute_loss_D_Z_CE(D_Z, Z2_trans, label_fake_CE)

            # loss_adv_Z_1 = lambda_adv_feat * D_Z.compute_loss(Z1, fake)
            # loss_adv_Z_2 = lambda_adv_feat * D_Z.compute_loss(Z2, valid)
            # loss_adv_Z_trans_1 = lambda_adv_feat * D_Z.compute_loss(Z1_trans, valid)
            # loss_adv_Z_trans_2 = lambda_adv_feat * D_Z.compute_loss(Z2_trans, valid)

            #Optional. Identity loss
            # loss_id_1 = lambda_id * criterion_recon(X1_trans, X1)
            # loss_id_2 = lambda_id * criterion_recon(X2_trans, X2)
            loss_id_1 = lambda_id * criterion_recon(X2_trans, X1)
            loss_id_2 = lambda_id * criterion_recon(X1_trans, X2)

            writer.add_scalar("Loss_G/loss_cycle_1", loss_cycle_1, global_i)
            writer.add_scalar("Loss_G/loss_cycle_2", loss_cycle_2, global_i)

            writer.add_scalar("Loss_G/loss_img_recon_1", loss_img_recon_1, global_i)
            writer.add_scalar("Loss_G/loss_img_recon_2", loss_img_recon_2, global_i)

            writer.add_scalar("Loss_G/loss_feat_recon_1", loss_feat_recon_1, global_i)
            writer.add_scalar("Loss_G/loss_feat_recon_2", loss_feat_recon_2, global_i)
            
            writer.add_scalar("Loss_G/loss_adv_1", loss_adv_1, global_i)
            writer.add_scalar("Loss_G/loss_adv_2", loss_adv_2, global_i)

            writer.add_scalar("Loss_G/loss_adv_Z_1", loss_adv_Z_1, global_i)
            writer.add_scalar("Loss_G/loss_adv_Z_2", loss_adv_Z_2, global_i)
            writer.add_scalar("Loss_G/loss_adv_Z_trans_1", loss_adv_Z_trans_1, global_i)
            writer.add_scalar("Loss_G/loss_adv_Z_trans_2", loss_adv_Z_trans_2, global_i)

            writer.add_scalar("Loss_G/loss_id_1", loss_id_1, global_i)
            writer.add_scalar("Loss_G/loss_id_2", loss_id_2, global_i)


            # loss_identity = lambda_id * ((loss_id_1 + loss_id_2) / 2)

            # Total loss
            loss_G = (
                loss_cycle_1 +
                loss_cycle_2 +
                loss_img_recon_1 +
                loss_img_recon_2 + 
                loss_feat_recon_1 + 
                loss_feat_recon_2 + 
                loss_adv_1 + 
                loss_adv_2 + 
                loss_adv_Z_1 + 
                loss_adv_Z_2 + 
                loss_adv_Z_trans_1 +
                loss_adv_Z_trans_2 +
                loss_id_1 + 
                loss_id_2
            )

            writer.add_scalar("Loss_G_tot/loss_G", loss_G, global_i)

            # Backprop
            loss_G.backward()
            optimizer_G.step()

            generator_update_counter = 0

        
        # Train Discriminators
        if discriminator_img_update_counter % opt.discriminator_img_rate == 0:
            # print("Discriminator train")

            """
            
            Train Image Domain Discriminators
            
            Forward:
            
            Loss:

            """

            # -----------------------
            #  Train Discriminator 1
            # -----------------------

            optimizer_D1.zero_grad()

            # loss_D1 = D1.compute_loss(X1, valid) + D1.compute_loss(X1_trans.detach(), fake)
            loss_D1 = compute_loss_D_adv(D1, X1, valid) + compute_loss_D_adv(D1, X1_trans.detach(), fake)
            writer.add_scalar("Loss_D/loss_D1", loss_D1, global_i)

            loss_D1.backward()
            optimizer_D1.step()

            # -----------------------
            #  Train Discriminator 2
            # -----------------------

            optimizer_D2.zero_grad()

            # loss_D2 = D2.compute_loss(X2, valid) + D2.compute_loss(X2_trans.detach(), fake)
            loss_D2 = compute_loss_D_adv(D2, X2, valid) + compute_loss_D_adv(D2, X2_trans.detach(), fake)
            writer.add_scalar("Loss_D/loss_D2", loss_D2, global_i)

            loss_D2.backward()
            optimizer_D2.step()

            discriminator_img_update_counter = 0

        if discriminator_feat_update_counter % opt.discriminator_feat_rate == 0:
                """
                
                Train Cross-domain Feature Consistency Discriminator
                
                Forward:
                
                Loss:

                """

                optimizer_D_Z.zero_grad()

                
                # loss_adv_Z_1 = lambda_adv_feat * compute_loss_D_Z(D_Z, Z1, label_Z1)
                # loss_adv_Z_2 = lambda_adv_feat * compute_loss_D_Z(D_Z, Z2, label_Z2)
                # loss_adv_Z_trans_1 = lambda_adv_feat * compute_loss_D_Z(D_Z, Z1_trans, label_Z1_trans)
                # loss_adv_Z_trans_2 = lambda_adv_feat * compute_loss_D_Z(D_Z, Z2_trans, label_Z2_trans)

                # Loss (Z1 : valid):
                if opt.D_Z_loss == 'BCE':
                    loss_D_Z = ( 
                        compute_loss_D_Z_BCE(D_Z, Z1.detach(), label_Z1_BCE) + 
                        compute_loss_D_Z_BCE(D_Z, Z2.detach(), label_Z2_BCE) + 
                        compute_loss_D_Z_BCE(D_Z, Z1_trans.detach(), label_Z1_trans_BCE) +
                        compute_loss_D_Z_BCE(D_Z, Z2_trans.detach(), label_Z2_trans_BCE)
                    )
                else:
                    loss_D_Z = ( 
                        compute_loss_D_Z_CE(D_Z, Z1.detach(), label_Z1_CE) + 
                        compute_loss_D_Z_CE(D_Z, Z2.detach(), label_Z2_CE) + 
                        compute_loss_D_Z_CE(D_Z, Z1_trans.detach(), label_Z1_trans_CE) +
                        compute_loss_D_Z_CE(D_Z, Z2_trans.detach(), label_Z2_trans_CE)
                    )
                # loss_D_Z = ( 
                #     D_Z.compute_loss(Z1.detach(), valid) + 
                #     D_Z.compute_loss(Z2_trans.detach(), fake) + 
                #     D_Z.compute_loss(Z2.detach(), fake) + 
                #     D_Z.compute_loss(Z1_trans.detach(), fake)
                # )
                writer.add_scalar("Loss_D/loss_D_Z", loss_D_Z, global_i)

                loss_D_Z.backward()
                optimizer_D_Z.step()
            
                discriminator_feat_update_counter = 0

        generator_update_counter += 1
        discriminator_img_update_counter += 1
        discriminator_feat_update_counter += 1

        # writer.add_graph(Enc1, X1)
        # writer.add_graph(Enc2, X2)
        # writer.add_graph(Dec1, Z1)
        # writer.add_graph(Dec2, Z2)
        # writer.add_graph(D1, X1)
        # writer.add_graph(D2, X2)
        # writer.add_graph(D_Z, Z1)

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        logger.info(
            "\r[Epoch %d/%d] [Batch %d/%d] [D_img loss: %f] [D_feat loss: %f] [G loss: %f] ETA: %s"
            % (epoch, opt.n_epochs, i, len(dataloader), (loss_D1 + loss_D2).item(), loss_D_Z.item(), loss_G.item(), time_left)
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done, global_i)

        global_i = global_i + 1

        writer.flush()


    epoch_end_time = time.time()
    epoch_time = epoch_end_time-epoch_start_time

    # Write to Tensorboard
    # writer.add_scalar("Loss_G/loss_cycle_1", loss_cycle_1, epoch)
    # writer.add_scalar("Loss_G/loss_cycle_2", loss_cycle_2, epoch)

    # writer.add_scalar("Loss_G/loss_img_recon_1", loss_img_recon_1, epoch)
    # writer.add_scalar("Loss_G/loss_img_recon_2", loss_img_recon_2, epoch)

    # writer.add_scalar("Loss_G/loss_feat_recon_1", loss_feat_recon_1, epoch)
    # writer.add_scalar("Loss_G/loss_feat_recon_2", loss_feat_recon_2, epoch)

    # writer.add_scalar("Loss_G/loss_adv_Z_1", loss_adv_Z_1, epoch)
    # writer.add_scalar("Loss_G/loss_adv_Z_2", loss_adv_Z_2, epoch)
    # writer.add_scalar("Loss_G/loss_adv_Z_trans_1", loss_adv_Z_trans_1, epoch)
    # writer.add_scalar("Loss_G/loss_adv_Z_trans_2", loss_adv_Z_trans_2, epoch)

    # writer.add_scalar("Loss_G/loss_id_1", loss_id_1, epoch)
    # writer.add_scalar("Loss_G/loss_id_2", loss_id_2, epoch)

    # writer.add_scalar("Loss_G/loss_G", loss_G, epoch)
    # writer.add_scalar("Loss_D/loss_D1", loss_D1, epoch)
    # writer.add_scalar("Loss_D/loss_D2", loss_D2, epoch)
    # writer.add_scalar("Loss_D/loss_D_Z", loss_D_Z, epoch)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D1.step()
    lr_scheduler_D2.step()
    lr_scheduler_D_Z.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(Enc1.state_dict(), "saved_models/%s/Enc1_%d.pth" % (opt.experiment_name, epoch))
        torch.save(Dec1.state_dict(), "saved_models/%s/Dec1_%d.pth" % (opt.experiment_name, epoch))
        torch.save(Enc2.state_dict(), "saved_models/%s/Enc2_%d.pth" % (opt.experiment_name, epoch))
        torch.save(Dec2.state_dict(), "saved_models/%s/Dec2_%d.pth" % (opt.experiment_name, epoch))
        torch.save(D1.state_dict(), "saved_models/%s/D1_%d.pth" % (opt.experiment_name, epoch))
        torch.save(D2.state_dict(), "saved_models/%s/D2_%d.pth" % (opt.experiment_name, epoch))
        torch.save(D_Z.state_dict(), "saved_models/%s/D_Z_%d.pth" % (opt.experiment_name, epoch))
    print("Epoch " + str(epoch) + " completed, took " + str(epoch_time) + " seconds to complete.")


writer.flush()