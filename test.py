"""

TODO List:
1. Load data
2. Load models
3. Run test with test set

"""

import torch
from torch.autograd import Variable
import argparse
import os
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
import skimage.io as io
from tifffile import imsave
# from PIL import Image
from models import *
import glob

parser = argparse.ArgumentParser()
# parser.add_argument("--folder_path", type=str, default='./dataset/Cytosolic2NLS_dataset_512_1x_1000_1000/test/A', required=False, help="Test image path")
# parser.add_argument("--save_path", type=str, default='images/20230504_Cytosolic2NLS_dataset_512_1x_1000_1000_featch5/195', required=False, help="Saving path")
# parser.add_argument("--enc1", type=str, default='./saved_models/20230504_Cytosolic2NLS_dataset_512_1x_1000_1000_featch5/Enc1_195.pth', required=False, help="Load state dictionary of encoder")
# parser.add_argument("--dec2", type=str, default='./saved_models/20230504_Cytosolic2NLS_dataset_512_1x_1000_1000_featch5/Dec2_195.pth', required=False, help="Load state dictionary of decoder" )
# parser.add_argument("--enc2", type=str, default='./saved_models/20230504_Cytosolic2NLS_dataset_512_1x_1000_1000_featch5/Enc2_195.pth', required=False, help="Load state dictionary of encoder")
# parser.add_argument("--dec1", type=str, default='./saved_models/20230504_Cytosolic2NLS_dataset_512_1x_1000_1000_featch5/Dec1_195.pth', required=False, help="Load state dictionary of decoder" )
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")

parser.add_argument("--dataset_name", type=str, default="Cytosolic2NLS_dataset_512_1x_1000_1000", help="name of the dataset")
parser.add_argument("--experiment_name", type=str, default="230527_CE_ratio2to5to1_200_img_adv6_feat2_Dz2_linear_corrloss", help="name of the experiment")
parser.add_argument("--direction", type=str, default="A2B", help='Direction of translation [A2B | B2A]')
parser.add_argument("--epoch", type=int, default=380, help="Epoch to test")

parser.add_argument("--in_channels", type=int, default=1, help="number of channels of input images")
parser.add_argument("--out_channels", type=int, default=1, help="number of channels of output images")
parser.add_argument("--feat_channels", type=int, default=2, help="number of channels of feature")

parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--test_count", type=int, default=30, help="number of test images")

parser.add_argument("--arch", type=str, default='AE', help="Architecture for Encoder/Decoder [AE | UNET]")
parser.add_argument("--unet_n_downsample", type=int, default=8, help="Number of downsampling for UNET") # 9 for 1x1 in 

parser.add_argument("--Enc_activation", type=str, default='relu', help="Output activation function for for Encoder/Decoder [tanh | relu]")
parser.add_argument("--Dec_activation", type=str, default='tanh', help="Output activation function for for Encoder/Decoder [tanh | relu]")

opt = parser.parse_args()
print(opt)


if opt.direction == "A2B":
    dataset_dir = os.path.join("./dataset", opt.dataset_name, 'test', 'A')
    enc1_path = os.path.join('./saved_models', opt.experiment_name, 'Enc1_'+str(opt.epoch)+'.pth')
    enc2_path = os.path.join('./saved_models', opt.experiment_name, 'Enc2_'+str(opt.epoch)+'.pth')
    dec1_path = os.path.join('./saved_models', opt.experiment_name, 'Dec1_'+str(opt.epoch)+'.pth')
    dec2_path = os.path.join('./saved_models', opt.experiment_name, 'Dec2_'+str(opt.epoch)+'.pth')
else:
    dataset_dir = os.path.join("./dataset", opt.dataset_name, 'test', 'B')
    enc1_path = os.path.join('./saved_models', opt.experiment_name, 'Enc2_'+str(opt.epoch)+'.pth')
    enc2_path = os.path.join('./saved_models', opt.experiment_name, 'Enc1_'+str(opt.epoch)+'.pth')
    dec1_path = os.path.join('./saved_models', opt.experiment_name, 'Dec2_'+str(opt.epoch)+'.pth')
    dec2_path = os.path.join('./saved_models', opt.experiment_name, 'Dec1_'+str(opt.epoch)+'.pth')


save_path = os.path.join('./images', opt.experiment_name, str(opt.epoch))

os.makedirs(save_path, exist_ok=True)
os.makedirs(f"{save_path}/png/", exist_ok=True)
os.makedirs(f"{save_path}/tif/", exist_ok=True)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Define model and load model checkpoint
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



# Enc1 = Encoder(in_channels=opt.in_channels, feat_channels=opt.feat_channels, dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual)
# Dec1 = Decoder(feat_channels=opt.feat_channels, out_channels=opt.in_channels, n_residual=opt.n_residual)
# Enc2 = Encoder(in_channels=opt.out_channels, feat_channels=opt.feat_channels, dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual)
# Dec2 = Decoder(feat_channels=opt.feat_channels, out_channels=opt.out_channels, n_residual=opt.n_residual)

Enc1 = nn.DataParallel(Enc1, device_ids = [0,1,2])
Enc1.to(device)
Dec2 = nn.DataParallel(Dec2, device_ids = [0,1,2])
Dec2.to(device)

Enc2 = nn.DataParallel(Enc2, device_ids = [0,1,2])
Enc2.to(device)
Dec1 = nn.DataParallel(Dec1, device_ids = [0,1,2])
Dec1.to(device)


Enc1.load_state_dict(torch.load(enc1_path))
Enc2.load_state_dict(torch.load(enc2_path))
Dec1.load_state_dict(torch.load(dec1_path))
Dec2.load_state_dict(torch.load(dec2_path))
Enc1.eval()
Enc2.eval()
Dec1.eval()
Dec2.eval()

files = sorted(glob.glob(f'{dataset_dir}/*.*'))

for i in tqdm(range(opt.test_count)):

    A_img = io.imread(files[i]).astype(np.float32)

    A_img1 = torch.from_numpy(A_img)#.to(device)

    if len(A_img1.shape)==3:
        A_img2 = A_img1
    elif len(A_img1.shape)==2:
        A_img2 = A_img1.unsqueeze(2)

    A_img2 = A_img2.unsqueeze(0).numpy()

    A_img2 = A_img2/2047.5-1 #4095/2

    X1 = torch.from_numpy(A_img2).permute(0, 3, 1, 2)

    # Upsample image
    with torch.no_grad():
        # Generate samples
        Z1 = Enc1(X1)
        print(Z1.shape)
        X2_trans = Dec2(Z1)
        X1_recon = Dec1(Z1)

        Z2_trans = Enc2(X2_trans)
        X1_trans_recon = Dec1(Z2_trans)


    # Denormalize X12=tensor    
    X1 = X1[0].cpu().float().numpy()
    X1 = (np.transpose(X1, (1, 2, 0)) + 1) / 2.0 * 4095.0
    
    Z1 = Z1[0].cpu().float().numpy()
    # Z1 = (Z1 + 1) / 2.0 * 4095.0
    # Z1 = (np.transpose(Z1, (1, 2, 0)) + 1) / 2.0 * 4095.0
    # print(Z1.shape)

    X2_trans = X2_trans[0].cpu().float().numpy()
    X2_trans = (np.transpose(X2_trans, (1, 2, 0)) + 1) / 2.0 * 4095.0
    
    Z2_trans = Z2_trans[0].cpu().float().numpy()
    # Z2_trans = (Z2_trans + 1) / 2.0 * 4095.0
    # Z2_trans = (np.transpose(Z2_trans, (1, 2, 0)) + 1) / 2.0 * 4095.0
    
    X1_recon = X1_recon[0].cpu().float().numpy()
    X1_recon = (np.transpose(X1_recon, (1, 2, 0)) + 1) / 2.0 * 4095.0
    
    X1_trans_recon = X1_trans_recon[0].cpu().float().numpy()
    X1_trans_recon = (np.transpose(X1_trans_recon, (1, 2, 0)) + 1) / 2.0 * 4095.0

    # Save image
    fn = (files[i]).split("/")[-1][:-4]
    
    io.imsave(f"{save_path}/png/{fn}_X1.png", X1)
    # io.imsave(f"{save_path}/png/{fn}_Z1.png", Z1)
    io.imsave(f"{save_path}/png/{fn}_X1_recon.png", X1_recon)
    io.imsave(f"{save_path}/png/{fn}_X2_trans.png", X2_trans)
    # io.imsave(f"{save_path}/png/{fn}_Z2_trans.png", Z2_trans)
    io.imsave(f"{save_path}/png/{fn}_X1_trans_recon.png", X1_trans_recon)

    io.imsave(f"{save_path}/tif/{fn}_X1.tif", X1)
    imsave(f"{save_path}/tif/{fn}_Z1.tif", Z1)
    # io.imsave(f"{save_path}/tif/{fn}_Z1.tif", Z1)
    io.imsave(f"{save_path}/tif/{fn}_X1_recon.tif", X1_recon)
    io.imsave(f"{save_path}/tif/{fn}_X2_trans.tif", X2_trans)
    imsave(f"{save_path}/tif/{fn}_Z2_trans.tif", Z2_trans)
    # io.imsave(f"{save_path}/tif/{fn}_Z2_trans.tif", Z2_trans)
    io.imsave(f"{save_path}/tif/{fn}_X1_trans_recon.tif", X1_trans_recon)