import os
import random
import glob
import numpy as np
from skimage import io
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import tifffile
from tqdm.contrib.itertools import product
from tqdm import tqdm

class HnE_Dataset(Dataset):
    def __init__(self, dataset_main_directory, mode="train", load_to_memory=False, seed=None, augmentation=True, normalize="whole", crop_size=None):
        super(HnE_Dataset, self).__init__()

        if seed != None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        data_dir = os.path.join(dataset_main_directory, mode)
        self.files_A = glob.glob(os.path.join(data_dir, "A") + "/*")
        self.files_B = glob.glob(os.path.join(data_dir, "B") + "/*")
        random.shuffle(self.files_A)
        random.shuffle(self.files_B)
        
        self.mode = mode
        self.augmentation = augmentation
        self.normalize = normalize
        self.crop_size = crop_size
        self.load_to_memory = load_to_memory

        A_sum = 0.0
        A_sum_sq = 0.0
        A_n_samples = 0

        B_sum = 0.0
        B_sum_sq = 0.0
        B_n_samples = 0

        if self.load_to_memory:
            self.data_A = []
            self.data_B = []
        
        if self.normalize == "whole" or self.load_to_memory:
            for file_A in tqdm(self.files_A, desc=f"Loading {mode}ing data from domain 1..."):
                img_A = torch.from_numpy(io.imread(file_A)).float().permute(2, 0, 1)
                if img_A.shape[0] > 3:
                    img_A = img_A[:3, :, :]
                if self.normalize == "whole":
                    A_sum += torch.sum(img_A)
                    A_sum_sq += torch.sum(img_A ** 2)
                    A_n_samples += img_A.numel()
                if self.load_to_memory:
                    self.data_A.append(img_A)
            
            for file_B in tqdm(self.files_B, desc=f"Loading {mode}ing data from domain 2..."):
                img_B = torch.from_numpy(io.imread(file_B)).float().permute(2, 0, 1)
                if img_B.shape[0] > 3:
                    img_B = img_B[:3, :, :]

                if self.normalize == "whole":
                    B_sum += torch.sum(img_B)
                    B_sum_sq += torch.sum(img_B ** 2)
                    B_n_samples += img_B.numel()

                if self.load_to_memory:
                    self.data_B.append(img_B)

        if self.normalize == "whole":
            self.A_mean = A_sum / A_n_samples
            
            self.A_std = torch.sqrt(A_sum_sq / A_n_samples - self.A_mean ** 2)

            self.B_mean = B_sum / B_n_samples
            self.B_std = torch.sqrt(B_sum_sq / B_n_samples - self.B_mean ** 2)

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
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

        if self.load_to_memory:
            img_A = self.data_A[index % len(self.data_A)]
            img_B = self.data_B[random.randint(0, len(self.data_B) - 1)]
        else:
            img_A = torch.from_numpy(io.imread(self.files_A[index % len(self.files_A)])).float().permute(2, 0, 1)
            img_B = torch.from_numpy(io.imread(self.files_B[random.randint(0, len(self.files_B) - 1)])).float().permute(2, 0, 1)
            if img_A.shape[0] > 3:
                img_A = img_A[:3, :, :]
            if img_B.shape[0] > 3:
                img_B = img_B[:3, :, :]

        if self.normalize == "whole":
            img_A = (img_A - self.A_mean) / self.A_std
            img_B = (img_B - self.B_mean) / self.B_std
        elif self.normalize == "individual":
            img_A = (img_A - torch.mean(img_A)) / torch.std(img_A)
            img_B = (img_B - torch.mean(img_B)) / torch.std(img_B)

        if self.crop_size != None:
            random_crop = transforms.RandomCrop((self.crop_size, self.crop_size))
            img_A = random_crop(img_A)
            img_B = random_crop(img_B)

        if self.augmentation:
            img_A = self.random_rotate(img_A)
            img_A = self.random_flip(img_A)

            img_B = self.random_rotate(img_B)
            img_B = self.random_flip(img_B)

        # Convert img_B from RGB to grayscale
        # img_A = transforms.functional.rgb_to_grayscale(img_A)
        # img_B = transforms.functional.rgb_to_grayscale(img_B)

        return {"A": img_A, "B": img_B}
        

class ImageDataset(Dataset):
    def __init__(self, dataset_main_directory, mode="train", seed=None, augmentation=True,augmentation_consistency=False, randomize_intensity=True, normalize=0, crop_size=256, intensity_type="invert"):
        super(ImageDataset, self).__init__()

        self.mode = mode
        self.augmentation_consistency = augmentation_consistency

        if seed != None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        self.crop_size = crop_size
        self.augmentation = augmentation
        self.normalize = normalize
        self.randomize_intensity = randomize_intensity
        self.intensity_type = intensity_type

        self.data_dir = os.path.join(dataset_main_directory, mode)
        self.data_A = sorted(glob.glob(os.path.join(self.data_dir, "A") + "/*.*"))
        self.data_B = sorted(glob.glob(os.path.join(self.data_dir, "B") + "/*.*"))

        self.t_len = 999999999
        for data in self.data_A:
            with tifffile.TiffFile(data) as tiff:
                t_len = len(tiff.pages)
            if self.t_len > t_len:
                self.t_len = t_len
        for data in self.data_B:
            with tifffile.TiffFile(data) as tiff:
                t_len = len(tiff.pages)
            if self.t_len > t_len:
                self.t_len = t_len
        self.t_len -= 1
        
    def random_rotation_90_180_270_360(self, img, augmentation):
        # Randomly choose one of the rotations: 90, 180, 270, or 360 degrees
        rotation_angle = random.choice([90, 180, 270, 360])

        augmented_img = transforms.functional.rotate(img, rotation_angle)  # Apply the random rotation

        augmentation.append(('angle', rotation_angle))

        return img, augmented_img[0], augmentation
    
    def random_flip(self, img, augmentation):
        # Randomly apply horizontal or vertical flips
        if random.random() < 0.5:
            augmented_img = transforms.functional.hflip(img)
            augmentation.append(('hflip', 1))
        else:
            augmented_img = img
            augmentation.append(('hflip', 0))
        
        if random.random() < 0.5:
            augmented_img = transforms.functional.vflip(img)
            augmentation.append(('vflip', 1))
        else:
            augmented_img = img
            augmentation.append(('vflip', 0))

        return img, augmented_img[0], augmentation

    def __getitem__(self, index):

        data_A_path = self.data_A[index % len(self.data_A)]
        data_B_path = self.data_B[random.randint(0, len(self.data_B) - 1)]

        data = {}

        random_idx_A = random.randint(10, self.t_len-10)
        random_idx_B = random.randint(10, self.t_len-10)

        img_A_full = io.imread(data_A_path, plugin="tifffile", key=random_idx_A)
        img_B_full = io.imread(data_B_path, plugin="tifffile", key=random_idx_B)
        
        data["A"] = torch.from_numpy(np.clip(img_A_full, 0, None)).float().unsqueeze(0)
        data["B"] = torch.from_numpy(np.clip(img_B_full, 0, None)).float().unsqueeze(0)

        if self.normalize:            
            eps = 1e-6
            data["A_norm"] = np.nanpercentile(data["A"], 95) + eps
            data["B_norm"] = np.nanpercentile(data["B"], 95) + eps

            data["A"] = data["A"] / data["A_norm"]
            data["B"] = data["B"] / data["B_norm"]
        else:
            data["A_norm"] = 1.0
            data["B_norm"] = 1.0

        random_crop_A = transforms.RandomCrop((self.crop_size, self.crop_size))
        data["A"] = random_crop_A(data["A"])

        random_crop_B = transforms.RandomCrop((self.crop_size, self.crop_size))
        data["B"] = random_crop_B(data["B"])
        
        # Apply the combined transform
        if self.augmentation:
            _, img_A_rot, _ = self.random_rotation_90_180_270_360(data["A"].clone(), [])
            _, img_A_rot_flip, _ = self.random_flip(img_A_rot.unsqueeze(0).clone(), [])
            data["A"] = img_A_rot_flip.unsqueeze(0)                
            
            _, img_B_rot, _ = self.random_rotation_90_180_270_360(data["B"].clone(), [])
            _, img_B_rot_flip, _ = self.random_flip(img_B_rot.unsqueeze(0).clone(), [])
            data["B"] = img_B_rot_flip.unsqueeze(0)
        else:
            data["A"] = data["A"].unsqueeze(0)
            data["B"] = data["B"].unsqueeze(0)

        # Augmentation for consistency loss
        if self.augmentation_consistency:
            augmentation_A = []
            _, img_A_rot, augmentation_A = self.random_rotation_90_180_270_360(data["A"].clone().unsqueeze(0), augmentation_A)
            _, img_A_rot_flip, augmentation_A = self.random_flip(img_A_rot.unsqueeze(0).clone(), augmentation_A)

            augmentation_B = []
            _, img_B_rot, augmentation_B = self.random_rotation_90_180_270_360(data["B"].clone().unsqueeze(0), augmentation_B)
            _, img_B_rot_flip, augmentation_B = self.random_flip(img_B_rot.unsqueeze(0).clone(), augmentation_B)

            data["A_aug"] = img_A_rot_flip
            data["A_aug_info"] = augmentation_A

            data["B_aug"] = img_B_rot_flip
            data["B_aug_info"] = augmentation_B
        
        data["A_inv"] = (data["A"].max() - data["A"])

        return data

    def __len__(self):
        return len(self.data_A)
    

