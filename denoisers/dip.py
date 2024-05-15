import numpy as np
import torch
import torch.nn as nn

from tv import TV_denoiser

class DIP_TV(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.alpha = args.alpha
        self.iter_num = args.iter_num
        self.lam = args.lam
        self.dip_iter = args.dip_iter
        self.tv_denoiser = TV_denoiser(args)

    def forward(self, x):
        x = self.tv_denoiser(x)
        input = get_noise().to(x.device)
        model = UNet_noskip(n_channels=x.shape[0], n_classes=x.shape[0]).to(x.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        loss_fn = nn.MSELoss().to(x.device)

        for i in range(self.dip_iter):
            pred = model(input)
            optimizer.zero_grad()
            y_pred = A(Phi, pred)
            x_loss = loss_fn(pred, x)
            y_loss = loss_fn(y_pred, y)
            loss = x_loss + y_loss * rho / mu
            loss.backward()
            optimizer.step()



            




class UNet_noskip(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_noskip, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024)
        self.up1 = Up_noskip(1024, 512, bilinear)
        self.up2 = Up_noskip(512, 256, bilinear)
        self.up3 = Up_noskip(256, 128, bilinear)
        self.up4 = Up_noskip(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        return logits
    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up_noskip(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x):
        x = self.up(x)
        return self.conv(x)     


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        return self.tanh(self.conv(x))