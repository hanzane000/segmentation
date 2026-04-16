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


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([x2, x1], dim=1)
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
         # 作用：先提取浅层基础特征 F_base，保持轻量（stem_channels=16）

        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pool7 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
         # 作用：多尺度低频照明估计（3x3 和 7x7）

        self.prior_head = nn.Sequential(
            nn.Conv2d(stem_channels * 2, stem_channels, kernel_size=1, bias=False),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels, 1, kernel_size=1),
        )
        # 作用：由残差信息生成 1 通道湿痕先验图 prior_map


        self.fuse = nn.Sequential(
            nn.Conv2d(stem_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        # 作用：把拼接后的增强特征融合回 in_channels，最小改动对接原 UNet 主干

    def forward(self, x):
        f_base = self.base(x)   # 基础浅层特征

        l3 = self.pool3(f_base) # 多尺度低频光照分量
        l7 = self.pool7(f_base)

        r3 = f_base - l3        # 局部残差（保留正负符号，不取绝对值）
        r7 = f_base - l7

        prior_feat = torch.cat([r3, r7], dim=1)     # 拼接残差信息作为先验特征输入
        prior_map = torch.sigmoid(self.prior_head(prior_feat))  # 生成湿痕先验图，sigmoid 限制在 [0,1]

        f_prior = f_base * prior_map        # 残差式门控：用先验图对基础特征进行逐像素加权
        f_enh = torch.cat([f_base, f_prior], dim=1)
        return self.fuse(f_enh)


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
        x = self.stem(x)        # 新增：先做湿痕先验增强

        x1 = self.inc(x)        # 再进入原始 UNet 编码器
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
