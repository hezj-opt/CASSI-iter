import torch
import torch.nn as nn
import numpy as np
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr
from .hsi_decnn import hsicnn_denoiser

class DIP_HSI(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.dip_iter = args.dip_iter
        self.step = args.step
        self.hsi_denoiser = hsicnn_denoiser(args)

    def forward(self, x, Phi, y):
        x = self.hsi_denoiser(x)
        input = get_noise(x.shape).to(x.device)
        model = UNet_noskip(n_channels=x.shape[0]).to(x.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        loss_fn = nn.MSELoss().to(x.device)
        loss_min = np.inf
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        for i in range(self.dip_iter):
            pred = model(input)
            optimizer.zero_grad()
            y_pred = A(Phi, shift_torch(pred, step=self.step))
            x_loss = loss_fn(pred, x)
            y_loss = loss_fn(y_pred, y)
            loss = x_loss + y_loss
            loss.backward()
            optimizer.step()

            if i % 25 == 0:
                if y_loss.item() < loss_min:
                    loss_min = y_loss.item()
                    output = pred
                    print(f"iter:{i}, y_loss_min={y_loss.item()}")

            if i % 100 == 0:
                print(f"iter:{i}, x_loss={x_loss.item()}, y_loss={y_loss.item()}")

        return output.squeeze(0).detach()

def get_noise(data_size, noise_type='u', var=1./10):
    shape = [1, data_size[0], data_size[1], data_size[2]]
    net_input = torch.zeros(shape)
    if noise_type == 'u':
        net_input = net_input.uniform_()*var
    elif noise_type == 'n':
        net_input = net_input.normal_()*var
    else:
        assert False
    return net_input.float()

def shift_torch(inputs, step):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col+(nC-1)*step).to(inputs.device)
    for i in range(nC):
        output[:, i, :, i*step:i*step+col] = inputs[:, i,:,:]
    return output

def A(Phi, x):
    temp = x * Phi
    y = torch.sum(temp, 1)
    return y

""" Parts of the U-Net model """
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


class UNet_noskip(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(UNet_noskip, self).__init__()
        self.n_channels = n_channels
        # self.n_classes = n_classes
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
        self.outc = OutConv(64, n_channels)

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
    