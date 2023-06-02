import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
# from PIL import Image
import torchvision.transforms as transforms
import skimage.io as io
from skimage.color import gray2rgb
# from PIL import Image
# import torchvision.transforms.functional as TF

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train", unaligned=True, in_datatype="f32", out_datatype="f32"):
        # self.transform = transforms_
        self.unaligned = unaligned
        self.in_datatype = in_datatype
        self.out_datatype = out_datatype
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        # img_A = skio.open(self.files_A[index % len(self.files_A)])

        # if self.unaligned:
        #     img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        # else:
        #     img_B = Image.open(self.files_B[index % len(self.files_B)])

        # img_A = img_A[:1,:,:]
        # img_B = img_B[:1,:,:]

        # TODO: If A u32 If B blah

        A_path = self.files_A[index % len(self.files_A)]
        if self.unaligned:
            B_path = self.files_B[random.randint(0, len(self.files_B) - 1)]
        else: 
            B_path = self.files_B[index % len(self.files_B)]

        # A
        if self.in_datatype == "f32":
            A_img = io.imread(A_path).astype(np.float32)
            A_img1 = torch.from_numpy(A_img)
            if len(A_img1.shape)==3:
                A_img2 = A_img1.numpy()
            elif len(A_img1.shape)==2:
                A_img2 = A_img1.unsqueeze(2).numpy()
            A_img2 = A_img2/2047.5-1 #4095/2
            img_A = torch.from_numpy(A_img2).permute(2, 0, 1)
        elif self.in_datatype == "uint8":
            A_img = io.imread(A_path).astype(np.int32)
            if len(A_img.shape) < 3:
                A_img = gray2rgb(A_img)
            A_img1 = torch.from_numpy(A_img)
            if len(A_img1.shape)==3:
                A_img2 = A_img1.numpy()
            elif len(A_img1.shape)==2:
                A_img2 = A_img1.unsqueeze(2).numpy()
            A_img2 = A_img2/255 #4095/2
            img_A = torch.from_numpy(A_img2).permute(2, 0, 1)
            # img_A = Image.open(A_path).convert("RGB")
            # img_A = TF.to_tensor(img_A)
            # img_A = img_A[:3,:,:]


        # B
        if self.out_datatype == "f32":
            B_img = io.imread(B_path).astype(np.float32)
            B_img1 = torch.from_numpy(B_img)
            if len(B_img1.shape)==3:
                B_img2 = B_img1.numpy()
            elif len(B_img1.shape)==2:
                B_img2 = B_img1.unsqueeze(2).numpy()
            B_img2 = B_img2/2047.5-1
            img_B = torch.from_numpy(B_img2).permute(2, 0, 1)
        elif self.out_datatype == "uint8":
            B_img = io.imread(B_path).astype(np.int32)
            if len(B_img.shape) < 3:
                B_img = gray2rgb(B_img)
            B_img1 = torch.from_numpy(B_img)
            if len(B_img1.shape)==3:
                B_img2 = B_img1.numpy()
            elif len(B_img1.shape)==2:
                B_img2 = B_img1.unsqueeze(2).numpy()
            B_img2 = B_img2/255
            img_B = torch.from_numpy(B_img2).permute(2, 0, 1)
            # c = Image.open(B_path).convert("RGB")
            # img_B = TF.to_tensor(img_B)
            # img_B = img_B[:3,:,:]
            

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files_A)

    # def transform(self, img1, img2):
        # Resize
        # resize = transforms.Resize(size=(512, 512))
        # img1 = resize(img1)
        # img2 = resize(img2)

        # # Random crop
        # i, j, h, w = transforms.RandomCrop.get_params(
        #     img1, output_size=(256, 256))
        # img1 = TF.crop(img1, i, j, h, w)
        # img2 = TF.crop(img2, i, j, h, w)
       
        # # Random horizontal flipping
        # if random.random() > 0.5:
        #     img1 = TF.hflip(img1)
        #     img2 = TF.hflip(img2)

        # # Random vertical flipping
        # if random.random() > 0.5:
        #     img1 = TF.vflip(img1)
        #     img2 = TF.vflip(img2)

        # Transform to tensor
        # img1 = TF.to_tensor(img1)
        # img2 = TF.to_tensor(img2)
        # return img1, img2
