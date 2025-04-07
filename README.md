# STABLE: Spatial and Quantitative Information Preserving Biomedical Image-to-Image Translation

This repository contains the open-source code for the paper:

**"Preserving Spatial and Quantitative Information in Unpaired Biomedical Image-to-Image Translation"**

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Supported File Formats and Data Types](#supported-file-formats-and-data-types)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Argument Explanations](#argument-explanations)
- [Example: Calcium Imaging Dataset](#example-calcium-imaging-dataset)
- [Packaging as a Python Package](#packaging-as-a-python-package)
- [License](#license)

---

## Overview

STABLE is an unpaired image-to-image translation framework designed to preserve spatial and quantitative details across different biomedical imaging modalities. It achieves this by enforcing information consistency and employing dynamic, learnable upsampling operators. The repository includes code for data preparation, model training, evaluation, and inference. Detailed logging and checkpointing facilitate model development and analysis.

---

## Requirements

While not specific requirements, the code was tested using the folowing versions of the Python packages and dependencies:

- **Python**: 3.11.10
- **CUDA**: 12.4 (if using GPU)
- **PyTorch**: 2.3.0
- **NumPy**: 1.26.4
- **scikit-image**: 0.24.0
- **torchvision**: 0.20.1
- **tensorboard**: 2.18.0
- **tqdm**: 4.67.1

The list of Python packages and dependencies are specified in the [`requirements.txt`](requirements.txt) file.

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/NICALab/STABLE.git
   cd STABLE
   ```

2. **Install Dependencies**  
   You can install the required packages via pip:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, install the package directly via pip:
   ```bash
   pip install .
   ```

**(Optional) Setup a Virtual Environment**  
   It is recommended to use a Conda or virtualenv environment:
   ```bash
   conda create -n stable_env python=3.11
   conda activate stable_env
   pip install -r requirements.txt
   ```

---

## Supported File Formats and Data Types

STABLE has been tested on both **TIF** and **PNG** file formats. The code accepts the following types of data:
- **2D Grayscale**
- **2D RGB**
- **3D Grayscale**
- **3D RGB**

**Note:** You must specify the `dim_order` when running the code to define the order of the dimensions. The accepted orders are:
- `CHW` (Channel, Height, Width)
- `HWC` (Height, Width, Channel)
- `ZHW` (Depth/Time, Height, Width)
- `HWZ` (Height, Width, Depth/Time)
- `ZCHW` (Depth/Time, Channel, Height, Width)
- `CHWZ` (Channel, Height, Width, Depth/Time)

Here, **C** represents the color channels (e.g., RGB), and **Z** represents the third dimension (which can be the z-axis for volumetric data or the time axis for videos).

---

## Dataset Structure

STABLE expects the dataset to be organized in a specific folder structure. For training and evaluation, the base dataset directory should contain subdirectories for each mode (e.g., `train`, `test`). Within each mode directory, there must be two folders:
- **A**: Contains images from the input domain.
- **B**: Contains images from the target domain.

### Example Directory Layout

```
/path/to/base_dataset_dir/
├── train
│   ├── A
│   │   ├── image1.tif/png
│   │   ├── image2.tif/png
│   │   └── ...
│   └── B
│       ├── image1.tif/png
│       ├── image2.tif/png
│       └── ...
├── test
│   ├── A
│   │   ├── image1.tif/png
│   │   └── ...
│   └── B
│       ├── image1.tif/png
│       └── ...
```

For inference, provide a directory containing the images from the input domain to be translated. The script processes all images in the directory.

---

## Usage

The codebase supports training and inference through command-line interfaces. After installing the package, you can run the commands `stable_train` and `stable_infer`.

### Training

The training script [train.py](stable/train.py) accepts several arguments that configure data paths, training parameters, and model architecture. Below is a summary of the key arguments used in the Calcium Imaging example:

- `--base_dataset_dir`: Path to the base dataset directory (must include `train` and `test` folders with `A` and `B` subdirectories).
- `--output_dir`: Directory to save directories directories containing output files, checkpoints, model settings, and logs.
- `--exp_name`: Experiment name for logging.
- **Related to Loss Weights:**
  - `--lambda_adv`: Weight for adversarial loss.
  - `--lambda_info`: Weight for information consistency loss.
  - `--lambda_cyc`: Weight for cycle consistency loss.
  - `--lambda_cyc_growth_target`: Epoch to reach full cycle consistency weight based on sigmoid growth function (optional).
- **Learning Rates:**
  - `--lr_G`: Learning rate for the generator.
  - `--lr_D`: Learning rate for the discriminator.
- **Logging & Checkpointing:**
  - `--log_train_iter`: Iterations between logging training statistics.
  - `--log_val_epoch`: Number of epochs between validation runs.
  - `--checkpoint_epoch`: Number of epochs between saving model checkpoints.
- **Training Schedule:**
  - `--epoch_start`: Starting epoch.
  - `--epoch_end`: Ending epoch.
- **Data Loading & Preprocessing:**
  - `--n_in`: Number of input channels.
  - `--n_out`: Number of output channels.
  - `--batch_size`: Batch size for training.
  - `--n_cpu`: Number of worker threads for data loading.
  - `--patch_size`: Size of the image patches for training.
  - `--dim_order`: Dimension order of the input images (e.g., `ZHW`).
  - `--normalize`: Normalization method (e.g., `percentile`).
  - `--normalize_range`: Range used for normalization (e.g., `0 99`).
  - `--normalize_clip`: Whether to clip values during normalization.
  - `--eps`: Small constant to prevent division by zero.
- **Generator Architecture:**
  - `--n_info`: Number of latent info channels.
  - `--G_mid_channels`: List of mid channels for the generator network.
  - `--G_norm_type`: Normalization type in the generator (`batch`, `instance`, or `none`).
  - `--G_demodulated`: Flag to use demodulated convolutions in the generator.
  - `--enc_act`: Activation function for the encoder.
  - `--dec_act`: Activation function for the decoder.
  - `--momentum`: Momentum for batch normalization.
- **Discriminator Configuration:**
  - `--D_n_scales`: Number of scales in the discriminator.
  - `--D_n_layers`: Number of layers per scale in the discriminator.
  - `--D_ds_stride`: Downsampling stride for the discriminator.
  - `--D_norm_type`: Normalization type in the discriminator (`batch`, `instance`, or `none`).
- **General:**
  - `--device`: Device to use (`cuda` for training on GPU or `cpu` for training on CPU).
  - `--seed`: Random seed for reproducibility.

#### Training Command Line Example with Key Arguments

Replace the placeholder directories and experiment name with your own paths and identifiers:

```bash
python -m stable.train \
  --base_dataset_dir <BASE_DATASET_DIR> \
  --output_dir <OUTPUT_DIR> \
  --exp_name <EXPERIMENT_NAME> \
  --lambda_adv <LAMBDA_ADV> \
  --lambda_info <LAMBDA_INFO> \
  --lambda_cyc <LAMBDA_CYC> \
  --lambda_cyc_growth_target <LAMBDA_CYC_GROWTH_TARGET> \
  --log_train_iter <LOG_TRAIN_ITER> \
  --log_val_epoch <LOG_VAL_EPOCH> \
  --checkpoint_epoch <CHECKPOINT_EPOCH> \
  --epoch_end <EPOCH_END> \
  --batch_size <BATCH_SIZE> \
  --patch_size <PATCH_SIZE> \
  --dim_order <DIM_ORDER> \
  --normalize <NORMALIZE_METHOD> \
  --normalize_range <NORMALIZE_RANGE_START> <NORMALIZE_RANGE_END> \
  --n_in <NUM_INPUT_CHANNELS> \
  --n_out <NUM_OUTPUT_CHANNELS> \
  --n_info <NUM_INFO_CHANNELS> \
  --G_mid_channels <G_MID_CHANNELS> \
  --G_norm_type <G_NORM_TYPE> \
  --device <DEVICE> \
  --seed <SEED>
```

### Inference

The inference script [infer.py](stable/infer.py) uses a trained model to translate images. Key arguments include:

- `--inference_dir`: Path to the directory containing images for inference.
- `--output_dir`: Base output directory where experiment output directories are stored.
- `--exp_name`: Experiment name corresponding to the trained model to inference.
- `--result_dir`: Directory where the translated results will be saved.
- `--model_settings_path`: (Optional) Path to a JSON file with model settings, if not provided, automatically loads model settings file from `/output_dir/exp_name/` by default.
- `--test_epoch`: The epoch number of the saved model to use for inference.
- **Data Loading & Preprocessing:**
  - `--patch_size`, `--dim_order`, `--normalize`, `--normalize_range`, `--normalize_clip`, `--eps`
- **Model Architecture:**
  - `--n_in`, `--n_out`, `--n_info`, `--G_mid_channels`, `--G_norm_type`, `--G_demodulated`, `--enc_act`, `--dec_act`, `--momentum`
- **General:**
  - `--batch_size`, `--n_cpu`, `--device`, `--seed`

#### Inference Command Line Example with Key Arguments

Replace the placeholder directories and experiment name with your own paths and identifiers:

```bash
python -m stable.infer \
  --inference_dir <INFERENCE_DIR> \
  --output_dir <OUTPUT_DIR> \
  --exp_name <EXPERIMENT_NAME> \
  --dim_order <DIM_ORDER> \
  --result_dir <RESULT_DIR> \
  --test_epoch <TEST_EPOCH>
```

## Example: Calcium Imaging Dataset

To replicate the results from the paper on the calcium imaging dataset, first structure your dataset directory as follows:

```
/path/to/calcium/dataset/
├── train
│   ├── A
│   └── B
└── test
    ├── A
    └── B
```

And you wish to save results to `/path/to/results/`. Replace the paths and experiment names accordingly when running the commands.

**Training Example:**

```bash
python -m stable.train \
  --base_dataset_dir '/path/to/calcium/dataset/' \
  --output_dir '/path/to/train/outputs/' \
  --exp_name 'calcium_exp' \
  --lambda_adv 1 \
  --lambda_info 10 \
  --lambda_cyc 1 \
  --lambda_cyc_growth_target 5000 \
  --epoch_end 5000 \
  --log_train_iter 100 \
  --log_val_epoch 100 \
  --checkpoint_epoch 100 \
  --epoch_end 5000 \
  --batch_size 1 \
  --patch_size 256 \
  --dim_order 'ZHW' \
  --normalize 'percentile' \
  --normalize_range 0 99 \
  --n_in 1 \
  --n_out 1 \
  --n_info 8 \
  --G_mid_channels 8 16 32 \
  --G_norm_type 'none' \
  --device 'cuda' \
  --seed 0
```

**Inference Example:**

```bash
python -m stable.infer \
  --inference_dir '/path/to/calcium/inference/data/' \
  --output_dir '/path/to/train/outputs/' \
  --exp_name 'calcium_exp' \
  --dim_order 'ZHW' \
  --result_dir '/path/to/results' \
  --test_epoch 0
```

---

## Packaging as a Python Package

The repository is structured as a Python package. With the provided [`setup.py`](setup.py), you can install the package using pip:
```bash
pip install .
```

After installing, the command-line tools `stable_train` and `stable_infer` will be available globally (replacing `python -m stable.train` and `python -m stable.infer` from previous examples), allowing you to train and perform inference on any compatible dataset.

---

## License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for more details.
