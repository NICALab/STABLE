from torchvision.transforms import RandomCrop
import torchvision.transforms as transforms
from skimage import io
import torch
import os
import numpy as np
import torchvision.transforms.functional as TF

num_crops = 2

# img_transform = transforms.Compose([RandomCrop(512, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
#                                 #  transforms.RandomHorizontalFlip(p=0.5),
#                                 #  transforms.RandomVerticalFlip(p=0.5),
#                                  ])
# crop = RandomCrop(512, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')

# toImg = ToPILImage()
cytosolic_input_dir_S = r"D:\Josh\Cytosolic2NLS\data\Zebrafish_Confocal_Cytosol2nls\230304_Casper_jGCaMP8m_4dpf_normcorre_SUPPORT_BEAR_alpha1E03\S"
cytosolic_input_dir_L = r"D:\Josh\Cytosolic2NLS\data\Zebrafish_Confocal_Cytosol2nls\230304_Casper_jGCaMP8m_4dpf_normcorre_SUPPORT_BEAR_alpha1E03\L"
NLS_input_dir_S = r"D:\Josh\Cytosolic2NLS\data\Zebrafish_Confocal_Cytosol2nls\230304_Casper_H2BGCaMP6s_4dpf_normcorre_SUPPORT_BEAR_alpha1E03\S"
NLS_input_dir_L = r"D:\Josh\Cytosolic2NLS\data\Zebrafish_Confocal_Cytosol2nls\230304_Casper_H2BGCaMP6s_4dpf_normcorre_SUPPORT_BEAR_alpha1E03\L"
cytosolic_output_dir_S = "./BEAR_dataset/S/A/"
cytosolic_output_dir_L = "./BEAR_dataset/L/A/"
NLS_output_dir_S = "./BEAR_dataset/S/B/"
NLS_output_dir_L = "./BEAR_dataset/L/B/"
output_dirs = [cytosolic_output_dir_S, cytosolic_output_dir_L, NLS_output_dir_S, NLS_output_dir_L]
dirs = [(cytosolic_input_dir_S, cytosolic_output_dir_S, cytosolic_input_dir_L, cytosolic_output_dir_L), (NLS_input_dir_S, NLS_output_dir_S, NLS_input_dir_L, NLS_output_dir_L)]
for output_dir in output_dirs:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

for dir_tup in dirs:
    input_dir_S = dir_tup[0]
    output_dir_S = dir_tup[1]
    input_dir_L = dir_tup[2]
    output_dir_L = dir_tup[3]
    print(input_dir_S, output_dir_S, input_dir_L, output_dir_L)
    cnt = 0
    for dir_name_S, dir_name_L in zip(os.listdir(input_dir_S), os.listdir(input_dir_L)):
        input_dir_path_S = os.path.join(input_dir_S, dir_name_S)
        input_dir_path_L = os.path.join(input_dir_L, dir_name_L)
        print(input_dir_path_S, input_dir_path_L)
        for filename_S, filename_L in zip(os.listdir(input_dir_path_S), os.listdir(input_dir_path_L)):

            input_file_path_S = os.path.join(input_dir_path_S, filename_S)
            input_file_path_L = os.path.join(input_dir_path_L, filename_L)

            input_S = io.imread(input_file_path_S).astype(np.float32)
            input_S = input_S[:,int(2048*1/4):]
            input_S = torch.from_numpy(input_S)
            input_L = io.imread(input_file_path_L).astype(np.float32)
            input_L = input_L[:,int(2048*1/4):]
            input_L = torch.from_numpy(input_L)

            for _ in range(num_crops):
                i, j, h, w = transforms.RandomCrop.get_params(input_S, output_size=(512, 512))
                output_S = TF.crop(input_S, i, j, h, w)
                output_L = TF.crop(input_L, i, j, h, w)
                output_S = output_S.detach().numpy()
                output_L = output_L.detach().numpy()

                file_name_S = "S_" + str(cnt) + '.tif'
                file_path_S = os.path.join(output_dir_S, file_name_S)
                io.imsave(file_path_S, output_S)

                file_name_L = "L_" + str(cnt) + '.tif'
                file_path_L = os.path.join(output_dir_L, file_name_L)
                io.imsave(file_path_L, output_L)

                cnt = cnt + 1