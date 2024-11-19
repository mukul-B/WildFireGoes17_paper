# -*- coding: utf-8 -*-
"""
This script contains Autoencoder implementation in pytorch

Created on Sun Jul 26 11:17:09 2020

@modified by: mukul badhan
on Sun Jul 23 11:17:09 2022

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

random_seed = 1
torch.manual_seed(random_seed)


class Encoder(nn.Module):
    """
    Autoencoder trained to separate class distributions in the latent space
    This is the encoder portion
    """

    def __init__(self, in_features):
        super(Encoder, self).__init__()
        self.in_features = in_features

        self.conv1 = nn.Conv2d(in_features, 64, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)

        # self.fc1 = nn.Linear(self.in_features, 4096)
        # self.fc2 = nn.Linear(4096, 1024)
        # self.fc3 = nn.Linear(1024, n_latents)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), kernel_size=(2, 2))

        # x = F.pad(x, (1, 1, 1, 1))
        x = F.relu(self.conv5(x))
        return x


class Decoder(nn.Module):
    """
    Autoencoder trained to separate class distributions in the latent space
    This is the decoder portion
    """

    def __init__(self, in_features, outtype=None):
        super(Decoder, self).__init__()
        self.in_features = in_features
        if not outtype:
            self.outtype = "relu"
        else:
            self.outtype = outtype

        self.tconv1 = nn.ConvTranspose2d(in_features, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)

        self.tconv2 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)

        self.conv5 = nn.Conv2d(64, 1, kernel_size=(3, 3), padding=1)

        if self.outtype == 'both':
            self.conv5b = nn.Conv2d(64, 1, kernel_size=(3, 3), padding=1)

        # self.fc1 = nn.Linear(n_latents, 1024)
        # self.fc2 = nn.Linear(1024, 4096)
        # self.fc3 = nn.Linear(4096, self.in_features)

    def forward(self, x):
        x1 = F.relu(self.conv1(self.tconv1(x, output_size=(10, 128, 64, 64))))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(self.tconv2(x1, output_size=(10, 64, 128, 128))))
        x1 = F.relu(self.conv4(x1))

        if self.outtype == 'relu':
            x1 = F.relu(self.conv5(x1))
            return x1
        if self.outtype == 'sigmoid':
            x1 = torch.sigmoid(self.conv5(255*x1))
            return x1

        if self.outtype == 'both':

            x_seg = torch.sigmoid(self.conv5(255*x1))
            x_sup = F.relu(self.conv5b(x1))

            return x_sup, x_seg


class Autoencoder(nn.Module):
    def __init__(self,in_channels=3, out_channels=1, last_activation='relu'):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(256, last_activation)

    def forward(self, x):
        x = self.encoder.forward(x)
        x = self.decoder.forward(x)
        return x
    