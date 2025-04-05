import os
import random
import glob
import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm


class StableDataset(Dataset):
    def __init__(self, base_dataset_dir, mode="train", paired=False, patch_size=None, normalize="percentile", 
                 normalize_range=(0, 99), normalize_clip=False, seed=None, augmentation=True, dim_order="ZHW", eps=1e-7):
        assert mode in ["train", "test", "val"], "Mode should be 'train' or 'test'"
        assert normalize in [None, "none", "percentile", "range", "minmax", "zscore"], "Normalize should be None, 'percentile', 'range', 'minmax', or 'zscore'"
        assert dim_order in ["CHW", "HWC", "ZHW", "HWZ", "ZCHW", "CHWZ"], "dim_order should be 'ZHW', 'HWZ', 'ZCHW', or 'CHWZ'"
        
        self.paired = paired
        self.base_dataset_dir = base_dataset_dir
        self.mode = mode
        self.patch_size = patch_size
        self.normalize = normalize
        self.normalize_range = normalize_range
        self.normalize_clip = normalize_clip
        self.augmentation = augmentation
        self.dim_order = dim_order
        
        if seed != None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        self.eps = eps
        
        # Check if subdirectories exist
        if not os.path.exists(os.path.join(base_dataset_dir, mode)):
            raise FileNotFoundError(f"Directory {os.path.join(base_dataset_dir, mode)} for {mode}ing does not exist")
        if not os.path.exists(os.path.join(base_dataset_dir, mode, "A")):
            raise FileNotFoundError(f"Directory {os.path.join(base_dataset_dir, mode, 'A')} for input domain does not exist")
        if not os.path.exists(os.path.join(base_dataset_dir, mode, "B")):
            raise FileNotFoundError(f"Directory {os.path.join(base_dataset_dir, mode, 'B')} for target domain does not exist")
        
        self.data_dir = os.path.join(base_dataset_dir, mode)
        
        files_A = sorted([f for f in glob.glob(os.path.join(self.data_dir, "A") + "/*") if os.path.isfile(f)])
        files_B = sorted([f for f in glob.glob(os.path.join(self.data_dir, "B") + "/*") if os.path.isfile(f)])
        
        if paired and mode == "train":
            file_pairs = list(zip(files_A, files_B))
            random.shuffle(file_pairs)            
            if len(file_pairs) % 2 != 0:
                file_pairs = file_pairs[:-1]
            half = len(file_pairs) // 2
            files_A = file_pairs[:half]
            files_B = file_pairs[half:]
            self.files_A  = [a for a, _ in files_A]
            self.files_B = [b for _, b in files_B]
        else:
            self.files_A = files_A
            self.files_B = files_B
        
        self.data_A = []
        self.data_B = []
        
        for file_A in tqdm(self.files_A, desc=f"Loading {mode}ing data from domain 1..."):
            img = self.get_image(file_A) # 'ZCHW'
            self.data_A.append(img)
        
        for file_B in tqdm(self.files_B, desc=f"Loading {mode}ing data from domain 2..."):
            img = self.get_image(file_B) # 'ZCHW'
            self.data_B.append(img)
        
    def __len__(self):
        return max(len(self.data_A), len(self.data_B))
        
    def normalize_image(self, img):
        if self.normalize == "percentile":
            img_min = np.percentile(img, self.normalize_range[0])
            img_max = np.percentile(img, self.normalize_range[1])
            img = (img - img_min) / (img_max - img_min + self.eps)
        elif self.normalize == "range":
            img = (img - self.normalize_range[0]) / (self.normalize_range[1] -self.normalize_range[0] + self.eps)
        elif self.normalize == "minmax":
            img_min = np.min(img)
            img_max = np.max(img)
            img = (img - img_min) / (img_max - img_min + self.eps)
        elif self.normalize == "zscore":
            img_mean = np.mean(img)
            img_std = np.std(img)
            img = (img - img_mean) / (img_std + self.eps)
        if self.normalize_clip:
            img = np.clip(img, 0, 1)
        return img
        
    def get_image(self, file_path):
        img = io.imread(file_path).astype(np.float32)
        if self.normalize is not None:
            img = self.normalize_image(img).squeeze()
        img = torch.from_numpy(img)
        # "CHW", "HWC", "ZHW", "HWZ", "ZCHW", "CHWZ" --> 'ZCHW'
        if len(img.shape) == 3:
            if self.dim_order == "HWZ":
                img = img.permute(2, 0, 1).unsqueeze(1)
            elif self.dim_order == "ZHW":
                img = img.unsqueeze(1)
            elif self.dim_order == "CHW":
                img = img.unsqueeze(0)
            elif self.dim_order == "HWC":
                img = img.permute(2, 0, 1).unsqueeze(0)
        if len(img.shape) == 4:
            if self.dim_order == "CHWZ":
                img = img.permute(3, 0, 1, 2)
                
        return img

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
    
    def __getitem__(self, index):
        if self.mode == "train":
            img_A = self.data_A[index % len(self.data_A)]
            img_B = self.data_B[random.randint(0, len(self.data_B) - 1)]
        else:
            img_A = self.data_A[index % len(self.data_A)]
            img_B = self.data_B[index % len(self.data_B)]
        
        if self.mode == "train":
            random_idx_A = random.randint(0, img_A.shape[0]-1)
            img_A = img_A[random_idx_A, :, :]
            random_idx_B = random.randint(0, img_B.shape[0]-1)
            img_B = img_B[random_idx_B, :, :]
        else:
            img_A = img_A[0, :, :]
            img_B = img_B[0, :, :]
                
        if self.patch_size != None:
            if self.mode == "train":
                random_crop = transforms.RandomCrop((self.patch_size, self.patch_size))
                img_A = random_crop(img_A)
                random_crop = transforms.RandomCrop((self.patch_size, self.patch_size))
                img_B = random_crop(img_B)
            else:
                img_A = transforms.functional.center_crop(img_A, (self.patch_size, self.patch_size))
                img_B = transforms.functional.center_crop(img_B, (self.patch_size, self.patch_size))
        
        if self.mode == "train" and self.augmentation:
            img_A = self.random_rotate(img_A)
            img_A = self.random_flip(img_A)

            img_B = self.random_rotate(img_B)
            img_B = self.random_flip(img_B)
        
        return {"A": img_A, "B": img_B}
    

class StableInferenceDataset(Dataset):
    def __init__(self, base_dataset_dir, patch_size=None, normalize="percentile", 
                 normalize_range=(0, 99), normalize_clip=False, dim_order="ZHW", eps=1e-7):
        assert normalize in [None, "none", "percentile", "range", "minmax", "zscore"], "Normalize should be None, 'percentile', 'range', 'minmax', or 'zscore'"
        assert dim_order in ["CHW", "HWC", "ZHW", "HWZ", "ZCHW", "CHWZ"], "dim_order should be 'ZHW', 'HWZ', 'ZCHW', or 'CHWZ'"
        
        self.base_dataset_dir = base_dataset_dir
        self.patch_size = patch_size
        self.normalize = normalize
        self.normalize_range = normalize_range
        self.normalize_clip = normalize_clip
        self.dim_order = dim_order
            
        self.eps = eps
        
        self.data_dir = base_dataset_dir
        
        self.files = sorted([f for f in glob.glob(self.data_dir + "/*") if os.path.isfile(f)])
        
        self.data = []
        self.paths = []
        
        for file in tqdm(self.files, desc=f"Loading inference data..."):
            img = self.get_image(file)
            self.data.append(img)
            for _ in range(img.shape[0]):
                self.paths.append(file)
        
        self.data = torch.cat(self.data, dim=0)
        
        # Lengths of A and B should be the same
        assert len(self.data) == len(self.paths)
        
    def __len__(self):
        return len(self.data)
        
    def normalize_image(self, img):
        if self.normalize == "percentile":
            img_min = np.percentile(img, self.normalize_range[0])
            img_max = np.percentile(img, self.normalize_range[1])
            img = (img - img_min) / (img_max - img_min + self.eps)
        elif self.normalize == "range":
            img = (img - self.normalize_range[0]) / (self.normalize_range[0] - self.normalize_range[1] + self.eps)
        elif self.normalize == "minmax":
            img_min = np.min(img)
            img_max = np.max(img)
            img = (img - img_min) / (img_max - img_min + self.eps)
        elif self.normalize == "zscore":
            img_mean = np.mean(img)
            img_std = np.std(img)
            img = (img - img_mean) / (img_std + self.eps)
        if self.normalize_clip:
            img = np.clip(img, 0, 1)
        return img
        
    def get_image(self, file_path):
        img = io.imread(file_path).astype(np.float32)
        if self.normalize is not None:
            img = self.normalize_image(img).squeeze()
        img = torch.from_numpy(img)
        # "CHW", "HWC", "ZHW", "HWZ", "ZCHW", "CHWZ" --> 'ZCHW'
        if len(img.shape) == 3:
            if self.dim_order == "HWZ":
                img = img.permute(2, 0, 1).unsqueeze(1)
            elif self.dim_order == "ZHW":
                img = img.unsqueeze(1)
            elif self.dim_order == "CHW":
                img = img.unsqueeze(0)
            elif self.dim_order == "HWC":
                img = img.permute(2, 0, 1).unsqueeze(0)
        if len(img.shape) == 4:
            if self.dim_order == "CHWZ":
                img = img.permute(3, 0, 1, 2)
                
        return img
    
    def __getitem__(self, index):

        img = self.data[index]
        path = self.paths[index]

        return {"A": img, "path_A": path}