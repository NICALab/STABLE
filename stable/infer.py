import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from skimage import io
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from stable.stable_dataset import StableInferenceDataset
from stable.stable_model import StableModel
from stable.stable_trainer import StableTrainer

def normalize_tensor(tensor):
    # Clone the tensor to avoid modifying the original in-place
    tensor = tensor.clone()

    # Compute the minimum and maximum values from the tensor
    low = float(tensor.min())
    high = float(tensor.max())

    # Clamp the tensor to ensure values are within [low, high]
    tensor.clamp_(min=low, max=high)
    
    # Subtract the minimum and divide by the range (using a small epsilon to avoid division by zero)
    tensor.sub_(low).div_(max(high - low, 1e-7))
    
    return tensor

def infer(args):
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)        
    
    infer_dataset = StableInferenceDataset(
        args.inference_dir, patch_size=args.patch_size, normalize=args.normalize, 
        normalize_range=args.normalize_range, normalize_clip=args.normalize_clip, dim_order=args.dim_order, eps=args.eps
    )
    infer_dataloader = DataLoader(
        infer_dataset, batch_size=1, num_workers=args.n_cpu, shuffle=False, drop_last=False
    )

    print(f"Length of inference dataloader: {len(infer_dataloader)}")

    model = StableModel(
        n_in=args.n_in, n_out=args.n_out, n_info=args.n_info,
        G_mid_channels=args.G_mid_channels, G_norm_type=args.G_norm_type,
        G_demodulated=args.G_demodulated, enc_act=args.enc_act, dec_act=args.dec_act,
        momentum=args.momentum, device=args.device
    )
    
    if args.model_settings_path is not None:
        settings_json = json.load(open(args.model_settings_path))
    else:
        experiment_dir = os.path.join(args.output_dir, "experiments", args.exp_name)
        settings_save_file = os.path.join(experiment_dir, f"settings.json")
        settings_json = json.load(open(settings_save_file))
    model.load_settings(settings_json)
    
    model.load_state_dict(os.path.join(args.output_dir, "experiments", args.exp_name, "saved_models", f"model_{args.test_epoch}.pth"))
    
    X_12_full = []
    previous_filename = None
    
    for batch in tqdm(infer_dataloader, desc=f"Infering", leave=False):
        model.eval()

        # Input data
        X_1 = batch["A"].to(args.device)
        X_1_filename = batch["path_A"][0]

        X_12 = model.infer(X_1)
        
        # Check if we have moved to a new file (and not on the very first file)
        if previous_filename is not None and X_1_filename != previous_filename:

            testfile = os.path.splitext(os.path.basename(previous_filename))[0]
            
            X_12_stack = torch.stack(X_12_full, axis=0)            
            X_12_stack = normalize_tensor(X_12_stack)
            
            if args.dim_order == "CHW" or args.dim_order == "HWC" or args.dim_order == "ZCHW" or args.dim_order == "CHWZ":
                save_image(X_12_stack, os.path.join(args.result_dir, f"{testfile}_translated.tif"), normalize=True)
            else:
                io.imsave(os.path.join(args.result_dir, f"{testfile}_translated.tif"), X_12_stack.numpy())
                
            print(f"Saved translation and input for {testfile}")
            
            # Reset accumulators for the new file
            X_12_full = []
            
        # Append current batch outputs
        X_12_full.append(X_12.squeeze().detach().cpu())
        previous_filename = X_1_filename
        
    if X_12_full:

        testfile = os.path.splitext(os.path.basename(previous_filename))[0]
        
        X_12_stack = torch.stack(X_12_full, axis=0)            
        X_12_stack = normalize_tensor(X_12_stack)
        
        if args.dim_order == "CHW" or args.dim_order == "HWC" or args.dim_order == "ZCHW" or args.dim_order == "CHWZ":
            save_image(X_12_stack, os.path.join(args.result_dir, f"{testfile}_translated.tif"), normalize=True)
        else:
            io.imsave(os.path.join(args.result_dir, f"{testfile}_translated.tif"), X_12_stack.numpy())
            
        print(f"Saved translation and input for {testfile}")
            
    print("Inference complete")

def main():
    parser = argparse.ArgumentParser(description="STABLE Training Script")

    # Data and Output
    parser.add_argument("--inference_dir", type=str, required=True, help="Path to inference data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output and checkpoints")
    parser.add_argument("--exp_name", type=str, default="experiment", help="Experiment name for logging")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory to save translation results")
    parser.add_argument("--model_settings_path", type=str, default=None, help="Path to model settings file")
    
    parser.add_argument("--test_epoch", type=int, default=0, help="Epoch to test")

    # Data Loading
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training and validation")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of data loading workers")

    # Patch & Normalization
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size for training image crops")
    parser.add_argument("--dim_order", type=str, default="ZHW", choices=["CHW", "HWC", "ZHW", "HWZ", "ZCHW", "CHWZ"], help="Dimension order of the input images [CHW | HWC | ZHW | HWZ | ZCHW | CHWZ]")
    parser.add_argument("--normalize", type=str, default="percentile", choices=["none", "percentile", "range", "minmax", "zscore"], help="Normalization method [percentile | range | minmax | none]")
    parser.add_argument("--normalize_range", type=float, nargs=2, default=[0.0, 99.0], help="Normalization range for normalize = 'percentile' or 'range'")
    parser.add_argument("--normalize_clip", type=bool, default=True, help="Whether to clip during normalization")
    parser.add_argument("--eps", type=float, default=1e-7, help="Small constant to prevent division by zero")

    # Architecture
    parser.add_argument("--n_in", type=int, default=1, help="Number of input channels")
    parser.add_argument("--n_out", type=int, default=1, help="Number of output channels")
    parser.add_argument("--n_info", type=int, default=8, help="Number of latent info channels")
    parser.add_argument("--G_mid_channels", type=int, nargs="+", default=[64, 128, 256, 512, 1024], help="Mid channels for the generator")
    parser.add_argument("--G_norm_type", type=str, default="batch", choices=['batch', 'instance', 'none'], help="Normalization type in generator [batch | instance | none]")
    parser.add_argument("--G_demodulated", type=bool, default=True, help="Use demodulated convolutions in generator")
    parser.add_argument("--enc_act", type=str, default="relu", choices=['sigmoid', 'tanh', 'softmax', 'leakyrelu', 'relu'], help="Encoder activation function [none | sigmoid | tanh | softmax]")
    parser.add_argument("--dec_act", type=str, default="relu", choices=['sigmoid', 'tanh', 'softmax', 'leakyrelu', 'relu'], help="Decoder activation function [none | sigmoid | tanh | softmax]")
    parser.add_argument("--momentum", type=float, default=0.1, help="Momentum for batch normalization")

    # General
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()
    infer(args)

if __name__ == "__main__":
    main()
