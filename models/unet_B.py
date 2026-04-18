import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class SelectiveSkipFusion(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        self.enc_proj = nn.Conv2d(encoder_channels, encoder_channels, kernel_size=1, bias=False)
        self.dec_proj = nn.Conv2d(decoder_channels, encoder_channels, kernel_size=1, bias=False)
        self.prior_proj = nn.Conv2d(1, encoder_channels, kernel_size=1, bias=False)

        self.gate_head = nn.Sequential(
            nn.Conv2d(encoder_channels, encoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels, encoder_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, encoder_feature, decoder_feature, prior_map):
        if decoder_feature.shape[2:] != encoder_feature.shape[2:]:
            decoder_feature = F.interpolate(
                decoder_feature,
                size=encoder_feature.shape[2:],
                mode="bilinear",
                align_corners=True,
            )

        if prior_map.shape[2:] != encoder_feature.shape[2:]:
            prior_map = F.interpolate(
                prior_map,
                size=encoder_feature.shape[2:],
                mode="bilinear",
                align_corners=True,
            )

        fused = self.enc_proj(encoder_feature) + self.dec_proj(decoder_feature) + self.prior_proj(prior_map)
        gate = self.gate_head(fused)
        encoder_refined = encoder_feature * (0.5 + gate)
        return encoder_refined


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.skip_fusion = SelectiveSkipFusion(in_channels // 2, in_channels // 2)

    def forward(self, x1, x2, prior_map):
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        refined_skip = self.skip_fusion(x2, x1, prior_map)
        x = torch.cat([refined_skip, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class WetnessPriorStem(nn.Module):
    def __init__(self, in_channels, stem_channels=16):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pool7 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)

        self.prior_head = nn.Sequential(
            nn.Conv2d(stem_channels * 2, stem_channels, kernel_size=1, bias=False),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels, 1, kernel_size=1),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(stem_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        f_base = self.base(x)

        l3 = self.pool3(f_base)
        l7 = self.pool7(f_base)

        r3 = f_base - l3
        r7 = f_base - l7

        prior_feat = torch.cat([r3, r7], dim=1)
        prior_map = torch.sigmoid(self.prior_head(prior_feat))

        f_prior = f_base * prior_map
        f_enh = torch.cat([f_base, f_prior], dim=1)
        enhanced = self.fuse(f_enh)

        return enhanced, prior_map


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, bilinear=True, base_c=32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.base_c = base_c
        self.stem = WetnessPriorStem(in_channels=in_channels, stem_channels=16)

        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.outc = OutConv(base_c, num_classes)

    def forward(self, x):
        x, prior = self.stem(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        p1 = F.interpolate(prior, size=x1.shape[2:], mode="bilinear", align_corners=True)
        p2 = F.interpolate(prior, size=x2.shape[2:], mode="bilinear", align_corners=True)
        p3 = F.interpolate(prior, size=x3.shape[2:], mode="bilinear", align_corners=True)
        p4 = F.interpolate(prior, size=x4.shape[2:], mode="bilinear", align_corners=True)

        x = self.up1(x5, x4, p4)
        x = self.up2(x, x3, p3)
        x = self.up3(x, x2, p2)
        x = self.up4(x, x1, p1)
        logits = self.outc(x)
        return logits
