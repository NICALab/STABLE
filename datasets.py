import os
import random
import glob
import numpy as np
from skimage import io
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import kornia

# class ImageDataset(Dataset):
#     def __init__(self, base_dataset_dir, mode, normalize='dataset', compute_stats=False, percentiles=[45.67070738302002, 76.07678560652094], augmentation=True, datatype='tif', seed=None, load_to_memory=True, size=(480, 480), test_idx=None):
#         assert mode in ['train', 'test'], "Mode should be 'train' or 'test'"
#         assert datatype in ['tif', 'png'], "Mode should be 'train' or 'test'"
#         assert normalize in ['dataset', 'data'], "Mode should be 'dataset' or 'data'"

#         self.test_idx = test_idx
#         self.datatype = datatype
#         self.size = size
#         self.augmentation = augmentation
#         self.normalize = normalize
#         self.load_to_memory = load_to_memory
#         self.compute_stats = compute_stats
#         self.percentiles = percentiles

#         self.eps = 1e-7

#         if seed != None:
#             torch.manual_seed(seed)
#             np.random.seed(seed)
#             random.seed(seed)
#             if torch.cuda.is_available():
#                 torch.backends.cudnn.deterministic = True
#                 torch.backends.cudnn.benchmark = False
#                 torch.cuda.manual_seed(seed)
#                 torch.cuda.manual_seed_all(seed)

#         data_dir = os.path.join(base_dataset_dir, mode)
#         if self.test_idx != None:
#             self.files_A = glob.glob(os.path.join(data_dir, "A") + "/*")[test_idx:test_idx+1]
#             self.files_B = glob.glob(os.path.join(data_dir, "B") + "/*")[test_idx:test_idx+1]
#         else:
#             self.files_A = glob.glob(os.path.join(data_dir, "A") + "/*")
#             self.files_B = glob.glob(os.path.join(data_dir, "B") + "/*")

#         if self.load_to_memory:
#             self.data_A = []
#             self.data_B = []
        
#         A_95_percentile = 0.0
#         B_95_percentile = 0.0

#         A_n_samples = 0
#         B_n_samples = 0

#         if self.normalize == "dataset" or self.load_to_memory:
#             for file_A in tqdm(self.files_A, desc=f"Loading {mode}ing data from domain 1..."):
#                 img_A = self.get_image(file_A)
                
#                 if self.compute_stats:
#                     A_95_percentile += np.percentile(img_A, 95)
#                     A_n_samples += 1
                    
#                 if self.load_to_memory:
#                     self.data_A.append(img_A)
                
#             for file_B in tqdm(self.files_B, desc=f"Loading {mode}ing data from domain 2..."):
#                 img_B = self.get_image(file_B)

#                 if self.compute_stats:
#                     B_95_percentile += np.percentile(img_B, 95)
#                     B_n_samples += 1

#                 if self.load_to_memory:
#                     self.data_B.append(img_B)

#         if self.compute_stats:
#             self.A_95_percentile = A_95_percentile / A_n_samples + self.eps
#             self.B_95_percentile = B_95_percentile / B_n_samples + self.eps
#         else:
#             self.A_95_percentile = self.percentiles[0]
#             self.B_95_percentile = self.percentiles[1]

#         if self.normalize == "dataset" and self.load_to_memory:
#             for i, _ in enumerate(tqdm(self.data_A, desc=f"Normalizing {mode}ing data from domain 1...")):
#                 self.data_A[i] = self.data_A[i] / self.A_95_percentile
            
#             for i, _ in enumerate(tqdm(self.data_B, desc=f"Normalizing {mode}ing data from domain 2...")):
#                 self.data_B[i] = self.data_B[i] / self.B_95_percentile
        
#         elif self.normalize == "data":
#             for i, _ in enumerate(tqdm(self.data_A, desc=f"Normalizing {mode}ing data from domain 1...")):
#                 self.data_A[i] = self.data_A[i] / ( np.percentile(self.data_A[i], 95 + self.eps) )
                
#             for i, _ in enumerate(tqdm(self.data_B, desc=f"Normalizing {mode}ing data from domain 2...")):
#                 self.data_B[i] = self.data_B[i] / ( np.percentile(self.data_B[i], 95 + self.eps) )

#     def __len__(self):
#         return max(len(self.files_A), len(self.files_B))

#     def random_rotate(self, img):
#         rotation_angle = random.choice([90, 180, 270, 360])
#         img = transforms.functional.rotate(img, rotation_angle)
#         return img

#     def random_flip(self, img):
#         if random.random() < 0.5:
#             img = transforms.functional.hflip(img)
#         if random.random() < 0.5:
#             img = transforms.functional.vflip(img)
#         return img
    
#     def get_image(self, file):
#         if self.datatype == 'png':
#             img = torch.from_numpy(io.imread(file)).float().permute(2, 0, 1) # [C, H, W]
#             if img.shape[0] > 3:
#                 img = img[:3, :, :]
#         else:
#             img = torch.from_numpy(io.imread(file)).float() # [T, H, W]
#             img = np.clip(img, 0, None)
#         return img

#     def __getitem__(self, index):

#         if self.load_to_memory:
#             img_A = self.data_A[index % len(self.data_A)]
#             img_B = self.data_B[random.randint(0, len(self.data_B) - 1)]
#         else:
#             img_A = self.get_image(self.files_A[index % len(self.files_A)])
#             img_B = self.get_image(self.files_B[random.randint(0, len(self.files_B) - 1)])

#             if self.normalize == "data":
#                 img_A = img_A / ( np.percentile(img_A, 95) + self.eps)
#                 img_B = img_B / ( np.percentile(img_B, 95) + self.eps)
#             elif self.normalize == "dataset":
#                 img_A = img_A / self.A_95_percentile
#                 img_B = img_B / self.B_95_percentile

#         # print(img_A.shape, img_B.shape)

#         if self.datatype == 'tif':
#             random_idx_A = random.randint(10, img_A.shape[0]-10)
#             random_idx_B = random.randint(10, img_B.shape[0]-10)

#             img_A = img_A[random_idx_A:random_idx_A+1, :, :]
#             img_B = img_B[random_idx_B:random_idx_B+1, :, :]

#         if self.size != None:
#             random_crop = transforms.RandomCrop((self.size, self.size))
#             img_A = random_crop(img_A)
#             img_B = random_crop(img_B)

#         if self.augmentation:
#             img_A = self.random_rotate(img_A)
#             img_A = self.random_flip(img_A)

#             img_B = self.random_rotate(img_B)
#             img_B = self.random_flip(img_B)

#         return {"A": img_A, "B": img_B} 
    

# # class ImageDataset_Infernce(Dataset):
# #     def __init__(self, base_dataset_dir, mode, normalize='dataset', compute_stats=False, percentiles=[45.67070738302002, 76.07678560652094], augmentation=True, datatype='tif', seed=None, load_to_memory=True, size=(480, 480), test_idx=None):
    





class ImageDataset(Dataset):
    def __init__(self, base_dataset_dir, mode, normalize=True, augmentation=True, datatype='tif', seed=101, size=(128, 128), test_idx=None, scale_ratio=(1.5, 1.0), equalize = (False, False)): # 1.0, 1.2
        assert mode in ['train', 'test'], "Mode should be 'train' or 'test'"
        assert datatype in ['tif', 'png'], "Mode should be 'train' or 'test'"
        # assert normalize in ['dataset', 'data'], "Mode should be 'dataset' or 'data'"

        self.test_idx = test_idx
        self.datatype = datatype
        self.size = size
        self.augmentation = augmentation
        self.normalize = normalize
        self.scale_ratio = scale_ratio
        self.equalize = equalize

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

        self.data_A = []
        self.data_B = []

        for file_A in tqdm(self.files_A, desc=f"Loading {mode}ing data from domain 1..."):
            img = self.get_image(file_A, 0, normalize, self.equalize[0])
            if len(img.shape) == 2:
                img = img.unsqueeze(0)
            if img.shape[1] >= self.size and img.shape[2] >= self.size:
                self.data_A.append(img)
            
        for file_B in tqdm(self.files_B, desc=f"Loading {mode}ing data from domain 2..."):
            img = self.get_image(file_B, 1, normalize, self.equalize[1])
            if len(img.shape) == 2:
                img = img.unsqueeze(0)
            if img.shape[1] >= self.size and img.shape[2] >= self.size:
                self.data_B.append(img)

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
    
    def substring_exists(self, s, sub):
        return sub in s
    
    def get_image(self, file, domain, normalize=True, equalize=True):
        if self.datatype == 'png':
            img = torch.from_numpy(io.imread(file, plugin="tifffile", key=list(range(0, 230, 4)))).float().permute(2, 0, 1) # [C, H, W]
            img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=(self.scale_ratio[domain], self.scale_ratio[domain]), mode='bilinear').squeeze()
            if img.shape[0] > 3:
                img = img[:3, :, :]
        else:
            # if not self.substring_exists(os.path.basename(file), "clahe"):
            #     img = torch.from_numpy(io.imread(file, plugin="tifffile", key=list(range(0, 230, 4)))).float() 
            # else:
            img = torch.from_numpy(io.imread(file)).float()
            if len(img.shape) == 2:
                img = img.unsqueeze(0)
            img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=(self.scale_ratio[domain], self.scale_ratio[domain]), mode='bilinear').squeeze()
            # if domain == 0:
            #     if self.substring_exists(file, "tau_zStack"):
            #         clip_lower = 1.6
            #     else:
            #         clip_lower = 450
            # else:
            #     if self.substring_exists(file, "tau_zStack"):
            #         clip_lower = 1.6
            #     else:
            #         clip_lower = 500
            img = np.clip(img, 0, None)
        if normalize:
            # Mean std
            # img = (img - img.mean()) / (img.std() + self.eps)
            # Clip values above 95th percentile 
            img = torch.clamp(img, 0, np.percentile(img, 99))
            # Max-Min Normalization
            img = (img - img.min()) / (img.max() - img.min() + self.eps)
            # img = img / ( np.percentile(img, 95) + self.eps)
            

        # if equalize:
        #     import kornia
        #     grid_size = int(48*(np.mean(img.shape)/795))
        #     img = kornia.enhance.equalize_clahe(img, clip_limit=10.0, grid_size=(grid_size, grid_size))
        #     img = (img - img.min()) / (img.max() - img.min() + self.eps)
        
        return img

    def __getitem__(self, index):

        img_A = self.data_A[index % len(self.data_A)]
        img_B = self.data_B[random.randint(0, len(self.data_B) - 1)]

        if self.datatype == 'tif':
            random_idx_A = random.randint(0, img_A.shape[0]-1)
            random_idx_B = random.randint(0, img_B.shape[0]-1)

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
        
        if self.equalize[0]:
            grid_size = 12
            img_A = kornia.enhance.equalize_clahe(img_A, clip_limit=10.0, grid_size=(grid_size, grid_size))
            img_A = (img_A - img_A.min()) / (img_A.max() - img_A.min() + self.eps)
        if self.equalize[1]:
            grid_size = 12
            img_B = kornia.enhance.equalize_clahe(img_B, clip_limit=10.0, grid_size=(grid_size, grid_size))
            img_B = (img_B - img_B.min()) / (img_B.max() - img_B.min() + self.eps)

        return {"A": img_A, "B": img_B} 



import zarr
import numpy as np
import torch

from torch.utils.data import Dataset


class Light2EM_Dataset(Dataset):
    def __init__(self, LM_data_path, EM_data_path, patch_size, overlap_size):
        self.LM_data = zarr.open(LM_data_path, mode='r')
        self.EM_data = zarr.open(EM_data_path, mode='r')
        self.patch_size = patch_size
        self.overlap_size = overlap_size

        print("LM shape: (z, y, x) = {}".format(self.LM_data.shape))
        print("EM shape: (z, y, x) = {}".format(self.EM_data.shape))

        self.LM_patch_indices = []
        self.EM_patch_indices = []
        for z in range(0, self.LM_data.shape[0], patch_size[0]-overlap_size[0]):
            for y in range(0, self.LM_data.shape[1], patch_size[1]-overlap_size[1]):
                for x in range(0, self.LM_data.shape[2], patch_size[2]-overlap_size[2]):
                    z_idx = z
                    y_idx = y
                    x_idx = x
                    if z+patch_size[0] > self.LM_data.shape[0]:
                        z_idx = self.LM_data.shape[0]-patch_size[0]
                    if y+patch_size[1] > self.LM_data.shape[1]:
                        y_idx = self.LM_data.shape[1]-patch_size[1]
                    if x+patch_size[2] > self.LM_data.shape[2]:
                        x_idx = self.LM_data.shape[2]-patch_size[2]
                    self.LM_patch_indices.append([z_idx, y_idx, x_idx])
                    if (x_idx != x) or x_idx == self.LM_data.shape[2]-patch_size[2]:
                        break
                if (y_idx != y) or y_idx == self.LM_data.shape[1]-patch_size[1]:
                    break
            if (z_idx != z) or z_idx == self.LM_data.shape[0]-patch_size[0]:
                break

        for z in range(0, self.EM_data.shape[0], patch_size[0]-overlap_size[0]):
            for y in range(0, self.EM_data.shape[1], patch_size[1]-overlap_size[1]):
                for x in range(0, self.EM_data.shape[2], patch_size[2]-overlap_size[2]):
                    z_idx = z
                    y_idx = y
                    x_idx = x
                    if z+patch_size[0] > self.EM_data.shape[0]:
                        z_idx = self.EM_data.shape[0]-patch_size[0]
                    if y+patch_size[1] > self.EM_data.shape[1]:
                        y_idx = self.EM_data.shape[1]-patch_size[1]
                    if x+patch_size[2] > self.EM_data.shape[2]:
                        x_idx = self.EM_data.shape[2]-patch_size[2]
                    self.EM_patch_indices.append([z_idx, y_idx, x_idx])
                    if x_idx != x:
                        break
                if y_idx != y:
                    break
            if z_idx != z:
                break
        
        print("Number of LM patches: {}".format(len(self.LM_patch_indices)))
        print("Number of LM patches: {}".format(len(self.EM_patch_indices)))
    
    def __len__(self):
        max_val = max(len(self.LM_patch_indices), len(self.EM_patch_indices))
        return max_val

    def __getitem__(self, idx):
        LM_idx = idx % len(self.LM_patch_indices)
        EM_idx = idx % len(self.EM_patch_indices)

        z, y, x = self.LM_patch_indices[LM_idx]
        LM_patch = self.LM_data[z:z+self.patch_size[0], y:y+self.patch_size[1], x:x+self.patch_size[2]]

        z, y, x = self.EM_patch_indices[EM_idx]
        EM_patch = self.EM_data[z:z+self.patch_size[0], y:y+self.patch_size[1], x:x+self.patch_size[2]]

        # change to torch tensor
        LM_patch = torch.from_numpy(LM_patch).to(torch.float32)
        EM_patch = torch.from_numpy(EM_patch).to(torch.float32)

        return {"A": LM_patch, "B": EM_patch}


# if __name__=="__main__":
#     dataset_3D = Light2EM_Dataset("./total_image.zarr", "./total_image_EM.zarr", [32, 512, 512], [16, 256, 256])
#     # dataset_2D = Light2EM_Dataset("./total_image.zarr", "./total_image_EM.zarr", [1, 512, 512], [0, 256, 256])

#     imgs = dataset[0]
#     LM_patch = imgs["A"]
#     EM_patch = imgs["B"]
#     print(LM_patch.shape)
#     print(max(LM_patch.flatten()))
#     print(EM_patch.shape)
#     print(max(EM_patch.flatten()))






# Main function
if __name__ == "__main__":
    base_dataset_dir = "/media/HDD1/josh/c2n/data/031924_c2n_select_256"
    mode = "train"
    normalize = True
    augmentation = True
    datatype = "tif"
    seed = 101
    size = 128
    test_idx = None
    dataset = ImageDataset(base_dataset_dir, mode, normalize, augmentation, datatype, seed, size, test_idx)
    print(len(dataset))
    print(dataset[0]["A"].shape, dataset[0]["B"].shape)
