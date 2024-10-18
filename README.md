# STABLE: Spatial and Quantitative Information Preserving Biomedical Image-to-Image Translation

This repository contains the code for the paper:

**"Preserving Spatial and Quantitative Information in Unpaired Biomedical Image-to-Image Translation"**

## Abstract

Analysis of biological samples often requires integrating diverse imaging modalities to gain a comprehensive understanding. While supervised biomedical image translation methods have shown success in synthesizing images across different modalities, they require paired data, which is often impractical to obtain due to challenges in data alignment and sample preparation. Unpaired methods, while not requiring paired data, struggle to preserve the precise spatial and quantitative information essential for accurate analysis.

To address these challenges, we introduce **STABLE** (Spatial and quanTitative informAtion preserving BiomedicaL imagE translation), an unpaired image-to-image translation method that emphasizes the preservation of spatial and quantitative information by enforcing information consistency and employing dynamic, learnable upsampling operators to achieve pixel-level accuracy. 

We validate STABLE across various biomedical imaging tasks, including translating calcium imaging data from zebrafish neurons and virtual histological staining, demonstrating its superior ability to preserve spatial details, signal intensities, and accurate alignment compared to existing methods.

## Software Specifications

The code has been tested with the following software versions:

- **CUDA**: 12.4
- **Python**: 3.11
- **PyTorch**: 2.3.0
- **NumPy**: 1.26.4
- **scikit-image**: 0.23.2

## Installation

1. Clone the repository:
git clone https://github.com/NICALab/STABLE.git

2. Navigate to the cloned folder:
cd ./STABLE

3. To train the model, run the following command template:
python train.py --exp_name 'EXPERIMENT_NAME' --output_dir 'PATH_TO_OUTPUT_DIRECTORY' --dataset_dir 'PATH_TO_DATASET_DIRECTORY' --data_type 'c2n' for calcium imaging translation task or 'stain' for virtual staining task
