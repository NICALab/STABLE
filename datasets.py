import os
import random
import glob
import numpy as np
from skimage import io
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, base_dataset_dir, mode, normalize=True, augmentation=True, seed=0, size=(128, 128), scale_ratio=(1.0, 1.0)):
        assert mode in ['train', 'test'], "Mode should be 'train' or 'test'"

        self.size = size
        self.augmentation = augmentation
        self.normalize = normalize
        self.scale_ratio = scale_ratio
        self.mode= mode

        self.eps = 1e-7

        if seed != None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        data_dir = os.path.join(base_dataset_dir, mode)
        self.files_A = glob.glob(os.path.join(data_dir, "A") + "/*.tif")
        self.files_B = glob.glob(os.path.join(data_dir, "B") + "/*.tif")

        self.data_A = []
        self.data_B = []

        for file_A in tqdm(self.files_A, desc=f"Loading {mode}ing data from domain 1..."):
            img = self.get_image(file_A, 0, normalize)
            if len(img.shape) == 2:
                img = img.unsqueeze(0)
            if img.shape[1] >= self.size and img.shape[2] >= self.size:
                self.data_A.append(img)
            
        for file_B in tqdm(self.files_B, desc=f"Loading {mode}ing data from domain 2..."):
            img = self.get_image(file_B, 1, normalize)
            if len(img.shape) == 2:
                img = img.unsqueeze(0)
            if img.shape[1] >= self.size and img.shape[2] >= self.size:
                self.data_B.append(img)

    def __len__(self):
        return max(len(self.data_A), len(self.data_B))

    def random_rotate(self, img):
        rotation_angle = random.choice([90, 180, 270, 360])
        img = transforms.functional.rotate(img, rotation_angle)
        return img

    def random_flip(self, img):
        if random.random() < 0.5:
            img = transforms.functional.hflip(img)
        if random.random() < 0.5:
            img = transforms.functional.vflip(img)
        return img
    
    def get_image(self, file, domain, normalize=True):
        img = torch.from_numpy(io.imread(file)).float()
        while len(img.shape) < 4:
            img = img.unsqueeze(0)
        img = torch.nn.functional.interpolate(img, scale_factor=(self.scale_ratio[domain], self.scale_ratio[domain]), mode='bilinear').squeeze()
        img = np.clip(img, 0, None)
        if normalize:
            img = torch.clamp(img, 0, np.percentile(img, 99))
            img = img / ( img.max() + self.eps )
        
        return img

    def __getitem__(self, index):

        img_A = self.data_A[index % len(self.data_A)]
        img_B = self.data_B[random.randint(0, len(self.data_B) - 1)]

        random_idx_A = random.randint(0, img_A.shape[0]-1)
        random_idx_B = random.randint(0, img_B.shape[0]-1)

        img_A = img_A[random_idx_A:random_idx_A+1, :, :]
        img_B = img_B[random_idx_B:random_idx_B+1, :, :]

        if self.size != None:
            if self.mode == "train":
                random_crop = transforms.RandomCrop((self.size, self.size))
                img_A = random_crop(img_A)
                img_B = random_crop(img_B)
            else:
                img_A = transforms.functional.center_crop(img_A, (self.size, self.size))
                img_B = transforms.functional.center_crop(img_B, (self.size, self.size))

        if self.augmentation:
            img_A = self.random_rotate(img_A)
            img_A = self.random_flip(img_A)

            img_B = self.random_rotate(img_B)
            img_B = self.random_flip(img_B)

        return {"A": img_A, "B": img_B} 


class HnEDataset(Dataset):
    def __init__(self, base_dataset_dir, mode, normalize=True, augmentation=True, seed=0, size=(128, 128),  scale_ratio=(1.0, 1.0), shuffle=False):
        assert mode in ['train', 'test'], "Mode should be 'train' or 'test'"
        self.mode = mode
        self.size = size
        self.augmentation = augmentation
        self.normalize = normalize
        self.scale_ratio = scale_ratio
        self.shuffle = shuffle

        if seed != None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        data_dir = os.path.join(base_dataset_dir, mode)
        self.files_A = sorted(glob.glob(os.path.join(data_dir, "A") + "/*"))
        self.files_B = sorted(glob.glob(os.path.join(data_dir, "B") + "/*"))

        self.data_A = []
        self.data_B = []

        for file_A in tqdm(self.files_A, desc=f"Loading {mode}ing data from domain 1..."):
            try:
                img = self.get_image_255_LivetoHnE(file_A, 0)
                if len(img.shape) == 2:
                    img = img.unsqueeze(0)
                if img.shape[-1] >= self.size or img.shape[-2] >= self.size:
                    self.data_A.append((img, file_A))
            except IOError:
                print("Image is truncated or corrupted.")
                continue
            
        for file_B in tqdm(self.files_B, desc=f"Loading {mode}ing data from domain 2..."):
            try:
                img = self.get_image_255_LivetoHnE(file_B, 1)
                if len(img.shape) == 2:
                    img = img.unsqueeze(0)
                if img.shape[-1] >= self.size or img.shape[-2] >= self.size:
                    self.data_B.append((img, file_B))
            except IOError:
                print("Image is truncated or corrupted.")
                continue

    def __len__(self):
        return max(len(self.data_A), len(self.data_B))

    def random_rotate(self, img):
        rotation_angle = random.choice([90, 180, 270, 360])
        img = transforms.functional.rotate(img, rotation_angle)
        return img

    def random_flip(self, img):
        if random.random() < 0.5:
            img = transforms.functional.hflip(img)
        if random.random() < 0.5:
            img = transforms.functional.vflip(img)
        return img
    
    def substring_exists(self, s, sub):
        return sub in s
    
    def get_image_255_LivetoHnE(self, file, domain):
        img = torch.from_numpy(io.imread(file)).to(torch.uint8)
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        if img.shape[-1] > 3 and img.shape[-1] < img.shape[-2]:
            img = img[:, :, :3]
        if img.shape[-1] < img.shape[-2]:
            img = img.permute(2,0,1)
        img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=(self.scale_ratio[domain], self.scale_ratio[domain]), mode='bilinear').squeeze()
        return img

    def __getitem__(self, index):

        img_A, path_A = self.data_A[index % len(self.data_A)]

        if self.mode == "train":
            if self.shuffle == True:
                img_B, path_B = self.data_B[random.randint(0, len(self.data_B) - 1)]
            else:
                img_B, path_B = self.data_B[index % len(self.data_B)]
        else:
            img_B, path_B = self.data_B[index % len(self.data_B)]

        img_A = img_A.float()
        img_B = img_B.float()
        if self.normalize:
            img_A = img_A / 255.0
            img_B = img_B / 255.0

        if self.size != None:
            if self.mode == "train":
                random_crop = transforms.RandomCrop((self.size, self.size))
                img_A = random_crop(img_A)
                img_B = random_crop(img_B)
            else:
                img_A = transforms.functional.center_crop(img_A, (self.size, self.size))
                img_B = transforms.functional.center_crop(img_B, (self.size, self.size))

        if self.augmentation:
            img_A = self.random_rotate(img_A)
            img_A = self.random_flip(img_A)

            img_B = self.random_rotate(img_B)
            img_B = self.random_flip(img_B)

        if self.mode == "train":
            return {"A": img_A, "B": img_B} 
        else:
            return {"A": img_A, "B": img_B, "path_A": path_A, "path_B": path_B}
