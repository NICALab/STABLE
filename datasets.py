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
    def __init__(self, base_dataset_dir, mode, normalize='dataset', augmentation=True, datatype='tif', seed=None, load_to_memory=True, size=(480, 480), test_idx=None):
        assert mode in ['train', 'test'], "Mode should be 'train' or 'test'"
        assert datatype in ['tif', 'png'], "Mode should be 'train' or 'test'"
        assert normalize in ['dataset', 'data'], "Mode should be 'dataset' or 'data'"

        self.test_idx = test_idx
        self.datatype = datatype
        self.size = size
        self.augmentation = augmentation
        self.normalize = normalize
        self.load_to_memory = load_to_memory

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
        if self.test_idx != None:
            self.files_A = glob.glob(os.path.join(data_dir, "A") + "/*")[test_idx:test_idx+1]
            self.files_B = glob.glob(os.path.join(data_dir, "B") + "/*")[test_idx:test_idx+1]
        else:
            self.files_A = glob.glob(os.path.join(data_dir, "A") + "/*")
            self.files_B = glob.glob(os.path.join(data_dir, "B") + "/*")

        A_sum = 0.0
        A_sum_sq = 0.0
        A_n_samples = 0

        B_sum = 0.0
        B_sum_sq = 0.0
        B_n_samples = 0

        if self.load_to_memory:
            self.data_A = []
            self.data_B = []

        # if self.data_type == 'tif':
        #     T, H, W = self.files_A.size
        #     C = 1
        # elif self.data_type == 'png':
        #     H, W, C = self.files_A.size

        if self.normalize == "dataset" or self.load_to_memory:
            for file_A in tqdm(self.files_A, desc=f"Loading {mode}ing data from domain 1..."):
                img_A = self.get_image(file_A)
                
                if self.normalize == "dataset":
                    A_sum += torch.sum(img_A)
                    A_sum_sq += torch.sum(img_A ** 2)
                    A_n_samples += img_A.numel()
                
                if self.load_to_memory:
                    self.data_A.append(img_A)
                
            for file_B in tqdm(self.files_B, desc=f"Loading {mode}ing data from domain 2..."):
                img_B = self.get_image(file_B)

                if self.normalize == "dataset":
                    B_sum += torch.sum(img_B)
                    B_sum_sq += torch.sum(img_B ** 2)
                    B_n_samples += img_B.numel()

                if self.load_to_memory:
                    self.data_B.append(img_B)

        if self.normalize == "dataset":
            self.A_mean = A_sum / A_n_samples
            self.A_std = torch.sqrt(A_sum_sq / A_n_samples - self.A_mean ** 2)

            for data_A in tqdm(self.data_A, desc=f"Normalizing {mode}ing data from domain 1..."):
                data_A = (data_A - self.A_mean) / self.A_std

            self.B_mean = B_sum / B_n_samples
            self.B_std = torch.sqrt(B_sum_sq / B_n_samples - self.B_mean ** 2)

            for data_B in tqdm(self.data_B, desc=f"Normalizing {mode}ing data from domain 2..."):
                data_B = (data_B - self.B_mean) / self.B_std
        
        elif self.normalize == "data":
            for data_A in tqdm(self.data_A, desc=f"Normalizing {mode}ing data from domain 1..."):
                data_A = (data_A - torch.mean(data_A)) / torch.std(data_A)

            for data_B in tqdm(self.data_B, desc=f"Normalizing {mode}ing data from domain 2..."):
                data_B = (data_B - torch.mean(data_B)) / torch.std(data_B)

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
    
    def get_image(self, file):
        if self.datatype == 'png':
            img = torch.from_numpy(io.imread(file)).float().permute(2, 0, 1) # [C, H, W]
            if img.shape[0] > 3:
                img = img[:3, :, :]
        else:
            img = torch.from_numpy(io.imread(file)).float() # [T, H, W]
        return img

    def __getitem__(self, index):

        if self.load_to_memory:
            img_A = self.data_A[index % len(self.data_A)]
            img_B = self.data_B[random.randint(0, len(self.data_B) - 1)]
        else:
            img_A = self.get_image(self.files_A[index % len(self.files_A)])
            img_B = self.get_image(self.files_B[random.randint(0, len(self.files_B) - 1)])

            if self.normalize == "data":
                img_A = (img_A - torch.mean(img_A)) / torch.std(img_A)
                img_B = (img_B - torch.mean(img_B)) / torch.std(img_B)
            elif self.normalize == "dataset":
                img_A = (img_A - self.A_mean) / self.A_std
                img_B = (img_B - self.B_mean) / self.B_std

        # print(img_A.shape, img_B.shape)

        if self.datatype == 'tif':
            random_idx_A = random.randint(10, img_A.shape[0]-10)
            random_idx_B = random.randint(10, img_B.shape[0]-10)

            img_A = img_A[random_idx_A:random_idx_A+1, :, :]
            img_B = img_B[random_idx_B:random_idx_B+1, :, :]

        if self.size != None:
            random_crop = transforms.RandomCrop((self.size, self.size))
            img_A = random_crop(img_A)
            img_B = random_crop(img_B)

        if self.augmentation:
            img_A = self.random_rotate(img_A)
            img_A = self.random_flip(img_A)

            img_B = self.random_rotate(img_B)
            img_B = self.random_flip(img_B)

        return {"A": img_A, "B": img_B} 