# CODE ADAPTED FROM AYOUB CHOUKRI

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


import matplotlib.pyplot as plt
import numpy as np
import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import imageio
# from PIL import Image

import logging
import os


# PARAMETERS
NOISE_STEPS = 1000
SEQUENCE_LENGHT = 210

BATCH_SIZE = 64
NUM_EPOCHS = 20
LR = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Used device: {device}")


# DIFFUSION MODEL
class Diffusion:
    def __init__(self, noise_steps=NOISE_STEPS, beta_start=0.0001, beta_end=0.02, sequence_lenght=SEQUENCE_LENGHT, device=device):
        """
        Initialize the Diffusion class.
       
        Attributes:
            noise_steps (int): Number of steps in the diffusion process.
            beta_start (float): Starting value of beta.
            beta_end (float): Ending value of beta.
            sequence_lenght (int): Size of the images.
            device (str): Device to run computations on.
            
            beta (torch.Tensor): Tensor containing linearly spaced beta values.
            alpha (torch.Tensor): Tensor containing corresponding alpha values.
            alpha_hat (torch.Tensor): Cumulative product of alpha values.
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.sequence_lenght = sequence_lenght
        self.device = device

        self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def to(self, device):
        """
        Move all model tensors to the specified device.
        """
        if device != self.device:
            if device == "cpu":
                logging.warning("Moving tensors to CPU. This might affect performance.")
            elif device == "cuda":
                logging.info("Moving tensors to CUDA")
                
            self.device = device
            self.beta = self.beta.to(device)
            self.alpha = self.alpha.to(device)
            self.alpha_hat = self.alpha_hat.to(device)
        else:
            logging.info("All tensors are already on the specified device: {}".format(device))
        return self

    def forward_diffusion(self, x, t):
        """
        Add noise to the images according to the timestep t.

        Args:
            x (torch.Tensor): Input sequences.
            t (torch.Tensor): Timestep values.

        Returns:
            tuple: Tuple containing the noised sequences and the noise.
        """
        
    
        t = t.flatten()
    
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t]).view(-1, 1, 1, 1)
        #sqrt_one_minus_alpha_hat = sqrt_one_minus_alpha_hat[:, None, None, None]
        noise = torch.normal(0, 1, x.shape, device=self.device)
        noised_sequences = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise

        return noised_sequences, noise

    def reverse_diffusion(self, model, x, start_t=None, num_images_to_return=None, test_noise=None):
        """
        Reverse diffusion process to generate sequences or observe steps from a specific t to t=0.

        Args:
            model: The model used to predict noise.
            x: Initial sequences or noise.
            start_t (int, optional): Initial timestep of reverse diffusion. Defaults to None (last step).
            num_sequences_to_return (int, optional): Number of additional sequences to return in addition to the final image (for visualisation). Defaults to None

        Returns:
            list: List of tuples containing reversed sequences and their corresponding timestep.
        """
        denoised_sample = []

        denoised_progression = []
        if num_sequences_to_return is None:
            num_sequences_to_return = 1
        if start_t is None:
            start_t = self.noise_steps - 1

        for i in reversed(range(1, start_t + 1)):
            noise = torch.normal(0, 1, x.shape, device=self.device)
            sqrt_alpha = torch.sqrt(self.alpha[i])
            one_minus_alpha = (1.0 - self.alpha[i])
            sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[i])
            predicted_noise = model.forward(x, torch.tensor([i], device=self.device))  
            x = (1/ sqrt_alpha) * (x - one_minus_alpha*predicted_noise/sqrt_one_minus_alpha_hat) + noise * torch.sqrt(self.beta[i])
            
            # We append the denoised images, if number of images to return is not specified
            # We only return the last image
            if num_images_to_return >= i:
                denoised_sample.append((x))
        
        return denoised_sample



# UNET ARCHIECTURE AND ASSOCIATED CLASSES
# class DoubleConv(nn.Module):
#     """ 
#     Double convolution layers with GroupNorm and GELU activation. Optionally includes a residual connection.

#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         mid_channels (int, optional): Number of channels in the intermediate layer. Defaults to None.
#         residual (bool, optional): Whether to include a residual connection. Defaults to False.
#     """
#     def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
#         super().__init__()

#         self.residual = residual
    
#         if not mid_channels:
#             mid_channels = out_channels

#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.GroupNorm(1, mid_channels),
#             nn.GELU(),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.GroupNorm(1, out_channels),
#         )

#     def forward(self, x):

#         if self.residual:
#             output = F.gelu(x + self.double_conv(x))
#         else:
#             output = self.double_conv(x)
            
#         return output


# class Down(nn.Module):
#     """ 
#     Downsampling layer with MaxPooling, DoubleConv, and temporal representation.

#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         emb_dim (int, optional): Dimension of the temporal embedding. Defaults to 256.
#     """
#     def __init__(self, in_channels, out_channels, emb_dim=256):
#         super().__init__()

#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, in_channels, residual=True),
#             DoubleConv(in_channels, out_channels),
#         )
#         self.emb_layer = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(emb_dim, out_channels),
#         )

#     def forward(self, x, t):

#         x = self.maxpool_conv(x)
#         emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
#         output = x + emb

#         return output


# class Up(nn.Module):
#     """ 
#     Upsampling layer with Upsample and DoubleConv. Also integrates temporal representation.

#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         emb_dim (int, optional): Dimension of the temporal embedding. Defaults to 256.
#     """
#     def __init__(self, in_channels, out_channels, emb_dim=256):
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         self.conv = nn.Sequential(
#             DoubleConv(in_channels, in_channels, residual=True),
#             DoubleConv(in_channels, out_channels, in_channels // 2),
#         )
#         self.emb_layer = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(emb_dim, out_channels),
#         )

#     def forward(self, x, skip_x, t):
        
#         x = self.up(x)
#         x = torch.cat([skip_x, x], dim=1)
#         x = self.conv(x)
#         emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
#         output = x + emb
    
#         return output


# class UNet(nn.Module):
#     """ 
#     Modular U-Net architecture.
#     """
#     def __init__(self, c_in=1, c_out=1, time_dim=256, device='cuda'):
#         super().__init__()
#         self.device = device
#         self.time_dim = time_dim

#         # Encoder

#         self.inc = DoubleConv(c_in, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 512)

#         # Decoder

#         self.up1 = Up(1024, 256)
#         self.up2 = Up(512, 128)
#         self.up3 = Up(256, 64)
#         self.up4 = Up(128, 64)

#         self.outc = nn.Conv2d(64, c_out, kernel_size=1)

#     def pos_encoding(self, t, channels):
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
#         pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
#         return pos_enc

#     def forward(self, x, t):
#         t = t.unsqueeze(-1).type(torch.float)
#         t = self.pos_encoding(t, self.time_dim)
        
#         x1 = self.inc(x)
#         x2 = self.down1(x1, t)
#         x3 = self.down2(x2, t)
#         x4 = self.down3(x3, t)
#         x5 = self.down4(x4, t)
        
#         x = self.up1(x5, x4, t)
#         x = self.up2(x, x3, t)
#         x = self.up3(x, x2, t)
#         x = self.up4(x, x1, t)

#         output = self.outc(x)
#         return output



# SINUSOIDAL EMBEDDING
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=device) * (-torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


# --- MLP ADAPTED TO VARIABLE SEQUENCE LENGHTS
class DiffusionMLP_Padded(nn.Module):
    def __init__(self, max_seq_len=210, n_channels=2, time_emb_dim=128, hidden_dim=512):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_channels = n_channels
        self.input_dim = max_seq_len * n_channels

        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
        )

    def forward(self, x, t):
        """
        x : (B, n_channels, seq_len) — seq_len ≤ max_seq_len
        t : (B,)
        """
        B, C, L = x.shape

        # padding (or shortening)
        if L < self.max_seq_len:
            pad = torch.zeros(B, C, self.max_seq_len - L, device=x.device)
            x = torch.cat([x, pad], dim=2)
        elif L > self.max_seq_len:
            x = x[:, :, :self.max_seq_len]

        x = x.view(B, -1)  # flatten
        t_emb = self.time_embed(t)
        xt = torch.cat([x, t_emb], dim=1)
        eps_pred = self.net(xt)
        return eps_pred.view(B, self.n_channels, self.max_seq_len)



# TRAIN LOOP
# def train(model, diffusion, device=device, epochs=NUM_EPOCHS, learning_rate=LR, dataloader=dataloader):
    
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

#     result_loss = []    

#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}")

#         for i, (images, _) in progress_bar:
#             images = images.to(device)
            
#             # Generate random diffusion steps for each image in the batch
#             t = torch.randint(0, diffusion.noise_steps, (images.shape[0],), device=device)

#             # Perform forward diffusion to obtain noised images and the true noise
#             noised_images, true_noise = diffusion.forward_diffusion(images, t)
#             #print(true_noise.shape)

#             optimizer.zero_grad()
            
#             # Forward pass through the model to predict noise
#             predicted_noise = model.forward(noised_images, t)
#             #print(predicted_noise.shape)

#             # Calculate the loss
#             loss_value = criterion(predicted_noise, true_noise)
#             #print(loss_value)

#             # Backward pass and optimize
#             loss_value.backward()
#             optimizer.step()
            
#             result_loss.append(loss_value.item())

#             running_loss += loss_value.item()
#             progress_bar.set_postfix(loss=running_loss / (i + 1))

#     return model, result_loss



# Define and move your model and diffusion to the device
# # model = UNet().to(device)
# model = DiffusionMLP(seq_len=200, n_channels=2).to(device)
# diffusion = Diffusion().to(device)

# # Train the model
# trained_model, rloss = train(model=model, diffusion=diffusion, device=device, epochs=15, learning_rate=LR)
