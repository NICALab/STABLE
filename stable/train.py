import argparse
from torch.utils.data import DataLoader
from stable.stable_dataset import StableDataset
from stable.stable_model import StableModel
from stable.stable_trainer import StableTrainer

def train(args):
    train_dataset = StableDataset(
        args.base_dataset_dir, mode="train", paired=False, patch_size=args.patch_size,
        normalize=args.normalize, normalize_range=args.normalize_range, normalize_clip=args.normalize_clip,
        seed=args.seed, augmentation=True, dim_order=args.dim_order, eps=args.eps
    )

    val_dataset = StableDataset(
        args.base_dataset_dir, mode="test", paired=False, patch_size=args.patch_size,
        normalize=args.normalize, normalize_range=args.normalize_range, normalize_clip=args.normalize_clip,
        seed=args.seed, augmentation=False, dim_order=args.dim_order, eps=args.eps
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.n_cpu, shuffle=True, drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.n_cpu, shuffle=False, drop_last=True
    )

    print(f"Length of train dataloader: {len(train_dataloader)}")
    print(f"Length of val dataloader: {len(val_dataloader)}")

    model = StableModel(
        n_in=args.n_in, n_out=args.n_out, n_info=args.n_info,
        G_mid_channels=args.G_mid_channels, G_norm_type=args.G_norm_type,
        G_demodulated=args.G_demodulated, enc_act=args.enc_act, dec_act=args.dec_act,
        momentum=args.momentum, D_n_scales=args.D_n_scales, D_n_layers=args.D_n_layers,
        D_ds_stride=args.D_ds_stride, D_norm_type=args.D_norm_type, device=args.device
    )

    trainer = StableTrainer(
        model=model, output_dir=args.output_dir, exp_name=args.exp_name,
        lambda_adv=args.lambda_adv, lambda_info=args.lambda_info, lambda_cyc=args.lambda_cyc,
        lambda_cyc_growth_target=args.lambda_cyc_growth_target, lr_G=args.lr_G, lr_D=args.lr_D,
        seed=args.seed, log_train_iter=args.log_train_iter, log_val_epoch=args.log_val_epoch,
        checkpoint_epoch=args.checkpoint_epoch
    )

    trainer.train(train_dataloader, val_dataloader, epoch_start=args.epoch_start, epoch_end=args.epoch_end)

def main():
    parser = argparse.ArgumentParser(description="STABLE Training Script")

    # Data and Output
    parser.add_argument("--base_dataset_dir", type=str, required=True, help="Path to base dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output and checkpoints")
    parser.add_argument("--exp_name", type=str, default="experiment", help="Experiment name for logging")

    # Loss Weights
    parser.add_argument("--lambda_adv", type=float, default=1, help="Weight for adversarial loss")
    parser.add_argument("--lambda_info", type=float, default=10, help="Weight for information consistency loss")
    parser.add_argument("--lambda_cyc", type=float, default=5, help="Weight for cycle consistency loss")
    parser.add_argument("--lambda_cyc_growth_target", type=int, default=None, help="Epoch to reach full lambda_cyc weight (optiona)")

    # Learning Rates
    parser.add_argument("--lr_G", type=float, default=3e-4, help="Learning rate for generator")
    parser.add_argument("--lr_D", type=float, default=3e-4, help="Learning rate for discriminator")

    # Logging and Checkpoints
    parser.add_argument("--log_train_iter", type=int, default=100, help="Iterations between logging training stats")
    parser.add_argument("--log_val_epoch", type=int, default=100, help="Epochs between validation runs")
    parser.add_argument("--checkpoint_epoch", type=int, default=100, help="Epochs between saving checkpoints")

    # Training Epochs
    parser.add_argument("--epoch_start", type=int, default=0, help="Start epoch")
    parser.add_argument("--epoch_end", type=int, default=5000, help="End epoch")

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

    # Discriminator
    parser.add_argument("--D_n_scales", type=int, default=1, help="Number of scales in discriminator")
    parser.add_argument("--D_n_layers", type=int, default=3, help="Number of layers per scale in discriminator")
    parser.add_argument("--D_ds_stride", type=int, default=2, help="Stride for downsampling in discriminator")
    parser.add_argument("--D_norm_type", type=str, default="batch", choices=['batch', 'instance', 'none'], help="Normalization type in discriminator [batch | instance | none]")

    # General
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
