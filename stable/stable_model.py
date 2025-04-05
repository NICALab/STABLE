import torch
import itertools

from stable.models import UNet, MultiDiscriminator

class StableModel(torch.nn.Module):
    def __init__(self, n_in, n_out, n_info, G_mid_channels=[64,128,256,512,1024], G_norm_type='batch', G_demodulated=True, enc_act='relu', dec_act='relu', momentum=0.1, D_n_scales=1, D_n_layers=3, D_ds_stride=2, D_norm_type='batch', device='cuda'):
        
        super().__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.n_info = n_info
        self.G_mid_channels = G_mid_channels
        self.G_norm_type = G_norm_type
        self.G_demodulated = G_demodulated
        self.enc_act = enc_act
        self.dec_act = dec_act
        self.momentum = momentum
        self.D_n_scales = D_n_scales
        self.D_n_layers = D_n_layers
        self.D_ds_stride = D_ds_stride
        self.D_norm_type = D_norm_type
        self.device = device
        
        self.Enc1 = UNet(n_in=n_in, n_out=n_info, mid_channels=G_mid_channels, norm_type=G_norm_type, 
                         demodulated=G_demodulated, act=enc_act, momentum=momentum).to(device)
        self.Dec1 = UNet(n_in=n_info, n_out=n_out, mid_channels=G_mid_channels, norm_type=G_norm_type, 
                         demodulated=G_demodulated, act=dec_act, momentum=momentum).to(device)
        self.Enc2 = UNet(n_in=n_in, n_out=n_info, mid_channels=G_mid_channels, norm_type=G_norm_type, 
                         demodulated=G_demodulated, act=enc_act, momentum=momentum).to(device)
        self.Dec2 = UNet(n_in=n_info, n_out=n_out, mid_channels=G_mid_channels, norm_type=G_norm_type, 
                         demodulated=G_demodulated, act=dec_act, momentum=momentum).to(device)
        
        self.D1 = MultiDiscriminator(channels=n_in, num_scales=D_n_scales, num_layers=D_n_layers, 
                                     downsample_stride=D_ds_stride, norm_type=D_norm_type, 
                                     kernel_size=4, stride=2, padding=1).to(device)
        self.D2 = MultiDiscriminator(channels=n_out, num_scales=D_n_scales, num_layers=D_n_layers, 
                                     downsample_stride=D_ds_stride, norm_type=D_norm_type, 
                                     kernel_size=4, stride=2, padding=1).to(device)
        
    def get_settings(self):
        return {'n_in': self.n_in, 
                'n_out': self.n_out, 
                'n_info': self.n_info, 
                'G_mid_channels': self.G_mid_channels, 
                'G_norm_type': self.G_norm_type, 
                'G_demodulated': self.G_demodulated, 
                'enc_act': self.enc_act, 
                'dec_act': self.dec_act, 
                'momentum': self.momentum, 
                'D_n_scales': self.D_n_scales, 
                'D_n_layers': self.D_n_layers, 
                'D_ds_stride': self.D_ds_stride, 
                'D_norm_type': self.D_norm_type, 
                'device': self.device}

    def eval(self):
        self.Enc1.eval()
        self.Dec1.eval()
        self.Enc2.eval()
        self.Dec2.eval()
        self.D1.eval()
        self.D2.eval()
        
    def train(self):
        self.Enc1.train()
        self.Dec1.train()
        self.Enc2.train()
        self.Dec2.train()
        self.D1.train()
        self.D2.train()
        
    def to(self, device):
        self.Enc1.to(device)
        self.Dec1.to(device)
        self.Enc2.to(device)
        self.Dec2.to(device)
        self.D1.to(device)
        self.D2.to(device)
        
    def load_state_dict(self, path):
        state_dict = torch.load(path)
        self.Enc1.load_state_dict(state_dict['Enc1'])
        self.Dec1.load_state_dict(state_dict['Dec1'])
        self.Enc2.load_state_dict(state_dict['Enc2'])
        self.Dec2.load_state_dict(state_dict['Dec2'])
        self.D1.load_state_dict(state_dict['D1'])
        self.D2.load_state_dict(state_dict['D2'])
        
    def save_state_dict(self, path):
        state_dict = {'Enc1': self.Enc1.state_dict(),
                      'Dec1': self.Dec1.state_dict(),
                      'Enc2': self.Enc2.state_dict(),
                      'Dec2': self.Dec2.state_dict(),
                      'D1': self.D1.state_dict(),
                      'D2': self.D2.state_dict()}
        torch.save(state_dict, path)
        
    def get_G_parameters(self):
        return itertools.chain(self.Enc1.parameters(), 
                               self.Dec1.parameters(), 
                               self.Enc2.parameters(), 
                               self.Dec2.parameters())
        
    def get_D1_parameters(self):
        return self.D1.parameters()
    
    def get_D2_parameters(self):
        return self.D2.parameters()
    
    def load_settings(self, settings):
        
        model_settings = {
            'n_in': settings['n_in'],
            'n_out': settings['n_out'],
            'n_info': settings['n_info'],
            'G_mid_channels': settings['G_mid_channels'],
            'G_norm_type': settings['G_norm_type'],
            'G_demodulated': settings['G_demodulated'],
            'enc_act': settings['enc_act'],
            'dec_act': settings['dec_act'],
            'momentum': settings['momentum'],
            'D_n_scales': settings['D_n_scales'],
            'D_n_layers': settings['D_n_layers'],
            'D_ds_stride': settings['D_ds_stride'],
            'D_norm_type': settings['D_norm_type']
        }
        self.__init__(**model_settings)
    
    def forward_G(self, X_1, X_2):        
        Z_1 = self.Enc1(X_1)
        Z_2 = self.Enc2(X_2)
        
        X_12 = self.Dec2(Z_1)
        X_21 = self.Dec1(Z_2)

        Z_12 = self.Enc2(X_12)
        Z_21 = self.Enc1(X_21)

        X_121 = self.Dec1(Z_12)
        X_212 = self.Dec2(Z_21)
        
        return Z_1, Z_2, X_12, X_21, Z_12, Z_21, X_121, X_212
    
    def forward_D1(self, X_1):
        return self.D1(X_1)
    
    def forward_D2(self, X_2):
        return self.D2(X_2)
    
    def infer(self, X_1):
        self.eval()
        
        X_12 = self.Dec2(self.Enc1(X_1))
        
        return X_12