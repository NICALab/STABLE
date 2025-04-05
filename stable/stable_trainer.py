import os
import json
import random
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

class StableTrainer:
    def __init__(self, model, output_dir, exp_name, lambda_adv, lambda_info, lambda_cyc, lambda_cyc_growth_target=None, lr_G=3e-4, lr_D=3e-4, seed=None, log_train_iter=1, log_val_epoch=1, checkpoint_epoch=1):
        
        if seed != None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        # Dump settings to json file
        settings = model.get_settings()
        settings = {**settings, 'lr_G': lr_G, 'lr_D': lr_D, 'seed': seed, 'lambda_adv': lambda_adv, 'lambda_info': lambda_info, 'lambda_cyc': lambda_cyc, 'lambda_cyc_growth_target': lambda_cyc_growth_target}
        
        # Set up directories
        self.experiment_dir = os.path.join(output_dir, "experiments", exp_name)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        self.saved_checkpoints_dir = os.path.join(self.experiment_dir, f"saved_models")
        if not os.path.exists(self.saved_checkpoints_dir):
            os.makedirs(self.saved_checkpoints_dir, exist_ok=True)
        
        settings_save_file = os.path.join(self.experiment_dir, f"settings.json")
        with open(settings_save_file, 'w') as file:
            json.dump(settings, file)
            
        writer_logdir = os.path.join(self.experiment_dir, f"runs")
        os.makedirs(writer_logdir, exist_ok=True)
        
        # Setup writer for tensorboard
        self.writer = SummaryWriter(log_dir=writer_logdir)
        self.writer.add_text('Options', str(settings), 0)
        
        self.model = model
        self.device = model.device
        
        self.optimizer_G = torch.optim.AdamW(
            self.model.get_G_parameters(),
            lr=lr_G
        )
        self.optimizer_D1 = torch.optim.AdamW(self.model.get_D1_parameters(), lr=lr_D)
        self.optimizer_D2 = torch.optim.AdamW(self.model.get_D2_parameters(), lr=lr_D)
        
        self.criterion_recon_cyc = nn.L1Loss()
        self.criterion_recon_com = nn.L1Loss()
        
        self.lambda_adv = lambda_adv
        self.lambda_info = lambda_info
        self.lambda_cyc = lambda_cyc
        self.lambda_cyc_growth_target = lambda_cyc_growth_target
        
        self.log_train_iter = log_train_iter
        self.log_val_epoch = log_val_epoch
        self.checkpoint_epoch = checkpoint_epoch
        
        self.batches_done = 0
        
    def load_state_dict_train(self, epoch):
        model_path = os.path.join(self.saved_checkpoints_dir, f"model_{epoch}.pth")
        self.model.load_state_dict(model_path)
        optimizer_path = os.path.join(self.saved_checkpoints_dir, f"optimizer_{epoch}.pth")
        self.load_state_dict_optimizer(optimizer_path)
        
    def save_state_dict(self, epoch):
        model_path = os.path.join(self.saved_checkpoints_dir, f"model_{epoch}.pth")
        self.model.save_state_dict(model_path)
        optimizer_path = os.path.join(self.saved_checkpoints_dir, f"optimizer_{epoch}.pth")
        self.save_state_dict_optimizer(optimizer_path)
        
    def load_state_dict_optimizer(self, path):
        checkpoint = torch.load(path)
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D1.load_state_dict(checkpoint['optimizer_D1'])
        self.optimizer_D2.load_state_dict(checkpoint['optimizer_D2'])
        self.batches_done = checkpoint['batches_done']
    
    def save_state_dict_optimizer(self, path):
        state_dict = {'optimizer_G': self.optimizer_G.state_dict(),
                      'optimizer_D1': self.optimizer_D1.state_dict(),
                      'optimizer_D2': self.optimizer_D2.state_dict(),
                      'batches_done': self.batches_done}
        torch.save(state_dict, path)
                
    def compute_loss_D1(self, x, gt):
        loss = 0
        output = self.model.forward_D1(x)
        for out in output:
            squared_diff = (out - gt) ** 2
            loss += torch.mean(squared_diff)
        return loss
    
    def compute_loss_D2(self, x, gt):
        loss = 0
        output = self.model.forward_D2(x)
        for out in output:
            squared_diff = (out - gt) ** 2
            loss += torch.mean(squared_diff)
        return loss
    
    def calculate_weight(self, iteration, target_iteration, a, b, rate):
        if target_iteration is None:
            return b
        else: 
            def sigmoid(x, rate):
                return 1 / (1 + np.exp(-rate * x))
            if iteration < target_iteration:
                normalized_iteration = 12 * (iteration / target_iteration) - 6
                sigmoid_value = sigmoid(normalized_iteration, rate)
                weight = a + (b - a) * sigmoid_value
            else:
                weight = b
            return weight
    
    def overlay_images(self, pred, gt):
        while(len(pred.shape) < 4):
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)

        _, _, h, w = pred.shape
        overlay_img = torch.zeros((1, 3, h, w))

        overlay_img[0, 0, :, :] = pred
        overlay_img[0, 1, :, :] = gt

        return overlay_img
    
    def log_images(self, epoch, mode, X_1, X_2, Z_1, Z_2, X_12, X_21, Z_12, Z_21, X_121, X_212):
        X_1_grid = make_grid(X_1, normalize=True)
        X_2_grid = make_grid(X_2, normalize=True)
        
        X_12_grid = make_grid(X_12, normalize=True)
        X_21_grid = make_grid(X_21, normalize=True)

        X_121_grid = make_grid(X_121, normalize=True)
        X_212_grid = make_grid(X_212, normalize=True)

        self.writer.add_image(f"01. Input->Output ({mode})/01. Real Input Image (X_1)", X_1_grid, epoch)
        for ch in range(Z_1.shape[1]):
            Z_1_common_grid = make_grid(Z_1[:,ch:ch+1,:,:], normalize=True)
            self.writer.add_image(f"01. Input->Output ({mode}) Feature/02_{ch}. Common Feature CH {ch} (Z_1_common)", Z_1_common_grid, epoch)
        self.writer.add_image(f"01. Input->Output ({mode})/03. Translated Output Image (X_12)", X_12_grid, epoch)
        for ch in range(Z_12.shape[1]):
            Z_12_common_grid = make_grid(Z_12[:,ch:ch+1,:,:], normalize=True)
            self.writer.add_image(f"01. Input->Output ({mode}) Feature/04_{ch}. Translated Common Feature CH {ch} (Z_12_common)", Z_12_common_grid, epoch)
        self.writer.add_image(f"01. Input->Output ({mode})/05. Cycle Reconstructed Input Image (X_121)", X_121_grid, epoch)

        self.writer.add_image(f"02. Output->Input ({mode})/01. Real Output Image (X_2)", X_2_grid, epoch)
        for ch in range(Z_2.shape[1]):
            Z_2_common_grid = make_grid(Z_2[:,ch:ch+1,:,:], normalize=True)
            self.writer.add_image(f"02. Output->Input ({mode}) Feature/02_{ch}. Common Feature CH {ch} (Z_2_common)", Z_2_common_grid, epoch)
        self.writer.add_image(f"02. Output->Input ({mode})/03. Translated Input Image (X_21)", X_21_grid, epoch)
        for ch in range(Z_21.shape[1]):
            Z_21_common_grid = make_grid(Z_21[:,ch:ch+1,:,:], normalize=True)
            self.writer.add_image(f"02. Output->Input ({mode}) Feature/04_{ch}. Translated Common Feature CH {ch} (Z_21_common)", Z_21_common_grid, epoch)
        self.writer.add_image(f"02. Output->Input ({mode})/05. Cycle Reconstructed Output Image (X_212)", X_212_grid, epoch)
        
        if self.model.n_in == 1 and self.model.n_out == 1:
            X_1_overlay = self.overlay_images(X_12[0], X_1[0])
            X_1_overlay_grid = make_grid(X_1_overlay, normalize=True)

            X_2_overlay = self.overlay_images(X_21[0], X_2[0])
            X_2_overlay_grid = make_grid(X_2_overlay, normalize=True)

            self.writer.add_image(f"01. Input->Output ({mode})/00. Overlay Input Image (X_12)", X_1_overlay_grid, epoch)
            self.writer.add_image(f"02. Output->Input ({mode})/00. Overlay Output Image (X_21)", X_2_overlay_grid, epoch)
        
    def train(self, train_dataloader, val_dataloader, epoch_start=0, epoch_end=99999):
        if epoch_start != 0:
            self.load_state_dict_train(epoch_start)
            print(f"Loaded model from epoch {epoch_start}")
        else:
            self.batches_done = 0
            
        # Adversarial ground truths
        valid = 1
        fake = 0
        
        for epoch in tqdm(range(epoch_start, epoch_end), desc="Training", position=0):
            
            for _, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False)):
                
                self.model.train()

                # Input data
                X_1 = batch["A"].to(self.device)
                X_2 = batch["B"].to(self.device)

                # Generator forward
                Z_1, Z_2, X_12, X_21, Z_12, Z_21, X_121, X_212 = self.model.forward_G(X_1, X_2)
                
                # Total Loss
                loss_G = 0.0

                # Image Domain Adversarial Losses
                loss_adv_1 = self.compute_loss_D1(X_21, valid)
                loss_G += self.lambda_adv * loss_adv_1

                loss_adv_2 = self.compute_loss_D2(X_12, valid)
                loss_G += self.lambda_adv * loss_adv_2

                # Common feature map reconstruction loss
                loss_com_rec_1 = self.criterion_recon_com(Z_12, Z_1.detach())
                loss_G += self.lambda_info * loss_com_rec_1

                loss_com_rec_2 = self.criterion_recon_com(Z_21.detach(), Z_2)
                loss_G += self.lambda_info * loss_com_rec_2

                # Image cycle reconstruction loss
                weight_cyc = self.calculate_weight(self.batches_done, 
                                                   self.lambda_cyc_growth_target*len(train_dataloader), 
                                                   0, self.lambda_cyc, 1)
                
                loss_img_cyc_1 = self.criterion_recon_cyc(X_121, X_1)
                loss_G += weight_cyc * loss_img_cyc_1

                loss_img_cyc_2 = self.criterion_recon_cyc(X_212, X_2)
                loss_G += weight_cyc * loss_img_cyc_2
                
                # Compute Losses
                self.optimizer_G.zero_grad()
                loss_G.backward()
                self.optimizer_G.step()
                
                # Train Input Domain Discriminator
                self.optimizer_D1.zero_grad()
                loss_D1 = (
                    self.compute_loss_D1(X_1, valid)
                    + self.compute_loss_D1(X_21.detach(), fake)
                )
                loss_D1.backward()
                self.optimizer_D1.step()

                # Train Output Domain Discriminator
                self.optimizer_D2.zero_grad()
                loss_D2 = (
                    self.compute_loss_D2(X_2, valid)
                    + self.compute_loss_D2(X_12.detach(), fake)
                )
                loss_D2.backward()
                self.optimizer_D2.step()

                if (self.batches_done + 1) % self.log_train_iter == 0:
                    self.writer.add_scalar("00.Overall (Train)/01. Total Generator Loss", loss_G, self.batches_done)
                    self.writer.add_scalar("00.Overall (Train)/02. Total Discriminator A Loss", loss_D1, self.batches_done)
                    self.writer.add_scalar("00.Overall (Train)/03. Total Discriminator B Loss", loss_D2, self.batches_done)

                    self.writer.add_scalar("01.Generator A->B (Train)/01. Image Domain Adversarial Loss A->B", loss_adv_2, self.batches_done)
                    self.writer.add_scalar("01.Generator A->B (Train)/02. Common Feature Reconstruction Loss A->B", loss_com_rec_1, self.batches_done)
                    self.writer.add_scalar("01.Generator A->B (Train)/03. Image Cycle Reconstruction Loss A->B", loss_img_cyc_1, self.batches_done)

                    self.writer.add_scalar("02.Generator B->A (Train)/01. Image Domain Adversarial Loss B->A", loss_adv_1, self.batches_done)
                    self.writer.add_scalar("02.Generator B->A (Train)/02. Common Feature Reconstruction Loss B->A", loss_com_rec_2, self.batches_done)
                    self.writer.add_scalar("02.Generator B->A (Train)/03. Image Cycle Reconstruction Loss B->A", loss_img_cyc_2, self.batches_done)
                    
                    self.log_images(self.batches_done, 'Train', X_1, X_2, Z_1, Z_2, X_12, X_21, Z_12, Z_21, X_121, X_212)
                    
                self.batches_done += 1
                
            if (epoch + 1) % self.log_val_epoch == 0:
                self.model.eval()
                with torch.no_grad():
                    batch = next(iter(val_dataloader))
                    
                    # Input data
                    X_1 = batch["A"].to(self.device)
                    X_2 = batch["B"].to(self.device)
                    
                    Z_1, Z_2, X_12, X_21, Z_12, Z_21, X_121, X_212 = self.model.forward_G(X_1, X_2)
                    self.log_images(epoch, 'Val', X_1, X_2, Z_1, Z_2, X_12, X_21, Z_12, Z_21, X_121, X_212)
                    
            if epoch % self.checkpoint_epoch == 0:
                self.save_state_dict(epoch)