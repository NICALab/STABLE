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
    def __init__(self, base_dataset_dir, mode, normalize='dataset', compute_stats=False, percentiles=[45.67070738302002, 76.07678560652094], augmentation=True, datatype='tif', seed=None, load_to_memory=True, size=(480, 480), test_idx=None):
        assert mode in ['train', 'test'], "Mode should be 'train' or 'test'"
        assert datatype in ['tif', 'png'], "Mode should be 'train' or 'test'"
        assert normalize in ['dataset', 'data'], "Mode should be 'dataset' or 'data'"

        self.test_idx = test_idx
        self.datatype = datatype
        self.size = size
        self.augmentation = augmentation
        self.normalize = normalize
        self.load_to_memory = load_to_memory
        self.compute_stats = compute_stats
        self.percentiles = percentiles

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
        if self.test_idx != None:
            self.files_A = glob.glob(os.path.join(data_dir, "A") + "/*")[test_idx:test_idx+1]
            self.files_B = glob.glob(os.path.join(data_dir, "B") + "/*")[test_idx:test_idx+1]
        else:
            self.files_A = glob.glob(os.path.join(data_dir, "A") + "/*")
            self.files_B = glob.glob(os.path.join(data_dir, "B") + "/*")

        if self.load_to_memory:
            self.data_A = []
            self.data_B = []
        
        A_95_percentile = 0.0
        B_95_percentile = 0.0

        A_n_samples = 0
        B_n_samples = 0

        if self.normalize == "dataset" or self.load_to_memory:
            for file_A in tqdm(self.files_A, desc=f"Loading {mode}ing data from domain 1..."):
                img_A = self.get_image(file_A)
                
                if self.compute_stats:
                    A_95_percentile += np.percentile(img_A, 95)
                    A_n_samples += 1
                    
                if self.load_to_memory:
                    self.data_A.append(img_A)
                
            for file_B in tqdm(self.files_B, desc=f"Loading {mode}ing data from domain 2..."):
                img_B = self.get_image(file_B)

                if self.compute_stats:
                    B_95_percentile += np.percentile(img_B, 95)
                    B_n_samples += 1

                if self.load_to_memory:
                    self.data_B.append(img_B)

        if self.compute_stats:
            self.A_95_percentile = A_95_percentile / A_n_samples + self.eps
            self.B_95_percentile = B_95_percentile / B_n_samples + self.eps
        else:
            self.A_95_percentile = self.percentiles[0]
            self.B_95_percentile = self.percentiles[1]

        if self.normalize == "dataset" and self.load_to_memory:
            for i, _ in enumerate(tqdm(self.data_A, desc=f"Normalizing {mode}ing data from domain 1...")):
                self.data_A[i] = self.data_A[i] / self.A_95_percentile
            
            for i, _ in enumerate(tqdm(self.data_B, desc=f"Normalizing {mode}ing data from domain 2...")):
                self.data_B[i] = self.data_B[i] / self.B_95_percentile
        
        elif self.normalize == "data":
            for i, _ in enumerate(tqdm(self.data_A, desc=f"Normalizing {mode}ing data from domain 1...")):
                self.data_A[i] = self.data_A[i] / ( np.percentile(self.data_A[i], 95 + self.eps) )
                
            for i, _ in enumerate(tqdm(self.data_B, desc=f"Normalizing {mode}ing data from domain 2...")):
                self.data_B[i] = self.data_B[i] / ( np.percentile(self.data_B[i], 95 + self.eps) )

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
            img = np.clip(img, 0, None)
        return img

    def __getitem__(self, index):

        if self.load_to_memory:
            img_A = self.data_A[index % len(self.data_A)]
            img_B = self.data_B[random.randint(0, len(self.data_B) - 1)]
        else:
            img_A = self.get_image(self.files_A[index % len(self.files_A)])
            img_B = self.get_image(self.files_B[random.randint(0, len(self.files_B) - 1)])

            if self.normalize == "data":
                img_A = img_A / ( np.percentile(img_A, 95) + self.eps)
                img_B = img_B / ( np.percentile(img_B, 95) + self.eps)
            elif self.normalize == "dataset":
                img_A = img_A / self.A_95_percentile
                img_B = img_B / self.B_95_percentile

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
    

# class ImageDataset_Infernce(Dataset):
#     def __init__(self, base_dataset_dir, mode, normalize='dataset', compute_stats=False, percentiles=[45.67070738302002, 76.07678560652094], augmentation=True, datatype='tif', seed=None, load_to_memory=True, size=(480, 480), test_idx=None):