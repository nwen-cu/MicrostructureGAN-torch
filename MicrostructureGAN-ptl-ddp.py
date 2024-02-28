import os
import sys
import socket

from math import sqrt

import torch
import torch.nn as nn

import lightning as ptl
from lightning.pytorch.loggers import NeptuneLogger

from torchinfo import summary

import neptune

import matplotlib.pyplot as plt

from tqdm.auto import trange
 
import numpy as np
import PIL

# exp = neptune.init_run(
#     project="",
# )

# exp["sys/tags"].add(["MicrostructureGAN", 'PT', 'DDP'])


# exp_id = exp['sys/id'].fetch()

# os.makedirs(f'saved_images/{exp_id}')
# os.makedirs(f'checkpoints/{exp_id}')


# Define image size
img_width = 256
img_height = 256
img_channels = 1
img_shape = (img_channels, img_height, img_width)

device = torch.device('cuda:0')

# Define hyperparameters
gp_coef = 1.
latent_dim = 100
# Scaled for DDP(2xGPU)
lr_d = 1e-4 #* sqrt(2)
lr_g = 2e-5 #* sqrt(2)

batch_size = 128



class LabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]
    
class MicrostructureImageDataModule(ptl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.prepare_data_per_node = False
        
    def prepare_data(self):
        # Load samples and rotate

        data_dir = 'training data'

        train_imgs = []
        train_labels = []
        labels = [0.73, 0.72, 0.7, 0.67, 0.66, 0.62, 0.56, 0.51]

        for i in range(4, 12):
            subset_imgs = []
            subset_label = labels[i - 4]
            for j in range(1, 6):
                img_dir = f'{data_dir}/{i}-{j}'
                for img_file in os.listdir(img_dir):
                    if img_file.startswith('.'): continue
                    img = PIL.Image.open(f'{img_dir}/{img_file}')
                    img_90 = img.transpose(PIL.Image.ROTATE_90)
                    img_180 = img.transpose(PIL.Image.ROTATE_180)
                    img_270 = img.transpose(PIL.Image.ROTATE_270)
                    arr = np.asarray(img)
                    arr_90 = np.asarray(img_90)
                    arr_180 = np.asarray(img_180)
                    arr_270 = np.asarray(img_270)
                    subset_imgs.append(arr)
                    subset_imgs.append(arr_90)
                    subset_imgs.append(arr_180)
                    subset_imgs.append(arr_270)
            if i != 9:
                train_imgs.append(subset_imgs)
                train_labels.append(subset_label * np.ones((len(subset_imgs), 1)))

        train_imgs = np.array(train_imgs).reshape((1080 * 7, 1, 256, 256)).astype(np.float32)
        train_imgs = (train_imgs.astype(np.float32) - 127.5) / 127.5
        train_labels = np.array(train_labels).reshape((1080 * 7, 1)).astype(np.float32)
        
        np.save('dataset_train_imgs.npy', train_imgs)
        np.save('dataset_train_labels.npy', train_labels)
        
    def setup(self, stage):
        self.dataset = LabeledImageDataset(np.load('dataset_train_imgs.npy'), 
                                           np.load('dataset_train_labels.npy'))
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


    
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.skip_conn = nn.Identity()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding='same'),
        )
        self.leakyrelu = nn.LeakyReLU(0.2)
        
    def forward(self, X):
        return self.leakyrelu(self.block(X) + self.skip_conn(X))



class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_noise = nn.Sequential(
            nn.Linear(latent_dim, 100 * 8 * 8),
            nn.Unflatten(1, (100, 8, 8)),
        )
        self.net_label = nn.Sequential(
            nn.Linear(1, 16 * 8 * 8),
            nn.Unflatten(1, (16, 8, 8)),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(116, 64, kernel_size=9, stride=4, padding=3, output_padding=1),
            
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            
            nn.Conv2d(64, 256, kernel_size=3, padding='same'),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.Upsample(scale_factor=2),
            
            nn.ConvTranspose2d(256, 128, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.LeakyReLU(0.2),
        
            nn.Conv2d(128, 1, kernel_size=11, stride=1, padding='same'),
            nn.Tanh(),
        )
        
        self.net_noise.apply(init_weights)
        self.net_label.apply(init_weights)
        self.net.apply(init_weights)
        
    def forward(self, noise: torch.Tensor, label: torch.Tensor):
        return self.net(torch.hstack([self.net_noise(noise), self.net_label(label)]))



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_img = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.net_label = nn.Sequential(
            nn.Linear(1, 64 * 64 * 20),
            nn.Unflatten(1, (20, 64, 64)),
        )
        self.net = nn.Sequential(
            # Original version
            nn.Conv2d(148, 512, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 256, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.2),

            ResBlock(64),

            nn.Flatten(),
            
            nn.Linear(65536, 512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 1),
        )
        
        self.net_img.apply(init_weights)
        self.net_label.apply(init_weights)
        self.net.apply(init_weights)
        
              
    def forward(self, img: torch.Tensor, label: torch.Tensor):
        return self.net(torch.hstack([self.net_img(img), self.net_label(label)]))


class MicrostructureGAN(ptl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.net_g = Generator()
        self.net_d = Discriminator()
        
    def forward(self, noises, labels):
        return self.net_g(noises, labels)
        
    def training_step(self, batch, batch_idx):        
        imgs_real, labels = batch

        batch_size = labels.size()[0]
        noises = torch.randn((batch_size, latent_dim)).type_as(labels)

        self._training_step_d(imgs_real, labels, noises)
        self._training_step_g(noises, labels)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.net_g.parameters(), lr=lr_g, betas=(0., .9), eps=1e-07)
        opt_d = torch.optim.Adam(self.net_d.parameters(), lr=lr_d, betas=(0., .9), eps=1e-07)
        return [opt_g, opt_d], []
    
    def on_train_epoch_end(self):
        if self.current_epoch  % 5 != 0: return
        r = 2
        c = 2
        noises = torch.rand((1, 100), device=self.device).repeat((4, 1))
        labels = torch.tensor([0.72, 0.7, 0.62, 0.51], device=self.device).reshape((4, 1))
        imgs_gen = self.net_g(noises, labels) * 127.5 + 127.5
        fig, axs = plt.subplots(r, c)
        idx = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(imgs_gen[idx, 0, :, :].detach().cpu().reshape((256, 256)), cmap='gray')
                axs[i, j].axis('off')
                idx += 1
        self.logger.experiment["generated_imgs"].append(fig, step=self.current_epoch )
        plt.close()
            
    
    def _training_step_d(self, imgs_real, labels, noises):
        _, opt_d = self.optimizers()
        
        self.toggle_optimizer(opt_d)
        
        opt_d.zero_grad()

        imgs_fake = self.net_g(noises, labels)
        loss_d_real = self.net_d(imgs_real, labels).mean()
        loss_d_fake = self.net_d(imgs_fake, labels).mean()

        grad_penalty = self._compute_gp(imgs_real, imgs_fake, labels)

        loss_d = loss_d_fake - loss_d_real + gp_coef * grad_penalty
        self.log("loss_d", loss_d, prog_bar=True)
        self.log("loss_d_fake", loss_d_fake)
        self.log("loss_d_real", loss_d_real)
        self.log("grad_penalty", grad_penalty)
        self.manual_backward(loss_d)

        opt_d.step()
        
        self.untoggle_optimizer(opt_d)

    def _training_step_g(self, noises, labels):
        opt_g, _ = self.optimizers()
        
        self.toggle_optimizer(opt_g)
        
        opt_g.zero_grad()

        imgs_gen = self.net_g(noises, labels)

        loss_g = -self.net_d(imgs_gen, labels).mean()
        self.log("loss_g", loss_g, prog_bar=True)
        self.manual_backward(loss_g)

        opt_g.step()
        
        self.untoggle_optimizer(opt_g)
        
    def _compute_gp(self, imgs_real, imgs_fake, labels):
        batch_size = labels.size()[0]

        epsilon = torch.rand((batch_size, 1, 1, 1)).type_as(labels).expand_as(imgs_real)
        imgs_interpolated = epsilon * imgs_real + (1 - epsilon) * imgs_fake
        imgs_interpolated.requires_grad_()

        logits_interpolated = self.net_d(imgs_interpolated, labels)
        grad_outputs = torch.ones_like(logits_interpolated)

        grad_interpolated = torch.autograd.grad(
            outputs=logits_interpolated,
            inputs=imgs_interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0].view(batch_size, -1)

        grad_norm = grad_interpolated.norm(2, 1)
        grad_penalty = ((grad_norm - 1) ** 2).mean()

        return grad_penalty
    
    
######################################################################




# def visualize(current_epoch):
#     r = 2
#     c = 2
#     noises = torch.rand((1, 100), device=device).repeat((4, 1))
#     labels = torch.tensor([0.72, 0.7, 0.62, 0.51], device=device).reshape((4, 1))
#     imgs_gen = net_g(noises, labels) * 127.5 + 127.5
#     fig, axs = plt.subplots(r, c)
#     idx = 0
#     for i in range(r):
#         for j in range(c):
#             axs[i, j].imshow(imgs_gen[idx, 0, :, :].detach().cpu().reshape((256, 256)), cmap='gray')
#             axs[i, j].axis('off')
#             idx += 1
#     exp["generated_imgs"].append(fig, step=current_epoch)
#     fig.savefig(f'saved_images/{exp_id}/{current_epoch}.png')
#     plt.close()
    
# def checkpoint(tag):
#     torch.save([net_g, net_d], f'checkpoints/{exp_id}/{tag}.pt')
#     exp[f'model_checkpoints/{tag}'].upload(f'checkpoints/{exp_id}/{tag}.pt')


dm = MicrostructureImageDataModule(batch_size)
model = MicrostructureGAN()
trainer = ptl.Trainer(
    accelerator="gpu",
    devices=2,
    max_epochs=700,
    strategy='ddp_find_unused_parameters_true',
    log_every_n_steps=1,
    logger=NeptuneLogger(project="pil-clemson/in-situ-test", prefix=''),
    
)
trainer.fit(model, dm)
