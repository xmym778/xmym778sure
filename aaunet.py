import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- HAAM模块 --------------------
def expend_as(tensor, rep):
    return tensor.expand(-1, rep, -1, -1)

class ChannelBlock(nn.Module):
    def __init__(self, filte):
        super(ChannelBlock, self).__init__()
        self.conv1 = nn.Conv2d(filte, filte, kernel_size=3, padding=3, dilation=3)
        self.conv2 = nn.Conv2d(filte, filte, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(filte)
        self.bn2 = nn.BatchNorm2d(filte)
        self.fc1 = nn.Linear(2 * filte, filte)
        self.fc2 = nn.Linear(filte, filte)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.Conv2d(2 * filte, filte, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(filte)

    def forward(self, x):
        a = self.relu(self.bn1(self.conv1(x)))
        b = self.relu(self.bn2(self.conv2(x)))
        gap = torch.mean(torch.cat([a, b], dim=1), dim=(2, 3))
        att = self.sigmoid(self.fc2(self.relu(self.fc1(gap))))
        att = att.unsqueeze(2).unsqueeze(3)
        y = a * att
        y1 = b * (1 - att)
        combined = torch.cat([y, y1], dim=1)
        out = self.relu(self.bn3(self.conv3(combined)))
        return out

class SpatialBlock(nn.Module):
    def __init__(self, filte, size):
        super(SpatialBlock, self).__init__()
        self.conv1 = nn.Conv2d(filte, filte, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(filte, filte, kernel_size=1)
        self.conv3 = nn.Conv2d(filte, 1, kernel_size=1)
        self.conv4 = nn.Conv2d(2 * filte, filte, kernel_size=size, padding=size // 2)
        self.bn1 = nn.BatchNorm2d(filte)
        self.bn2 = nn.BatchNorm2d(filte)
        self.bn3 = nn.BatchNorm2d(filte)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, ch_data):
        a = self.relu(self.bn1(self.conv1(x)))
        b = self.relu(self.bn2(self.conv2(a)))
        mix = self.relu(ch_data + b)
        att = self.sigmoid(self.conv3(mix))
        att_exp = expend_as(att, ch_data.size(1))
        y = ch_data * att_exp
        y1 = b * (1 - att_exp)
        combined = torch.cat([y, y1], dim=1)
        out = self.bn3(self.conv4(combined))
        return out

class HAAM(nn.Module):
    def __init__(self, filte, size):
        super(HAAM, self).__init__()
        self.channel_block = ChannelBlock(filte)
        self.spatial_block = SpatialBlock(filte, size)

    def forward(self, x):
        ch = self.channel_block(x)
        return self.spatial_block(x, ch)

# -------------------- Single-HAAM Conv Block --------------------
class AAUBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AAUBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.haam = HAAM(out_ch, size=3)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.haam(x)
        return x

# -------------------- Down / Up / Out --------------------
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = AAUBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.block(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
                nn.BatchNorm2d(in_ch // 2),
                nn.ReLU(inplace=True)
            )
            self.conv_ch = in_ch // 2
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv_ch = in_ch // 2

        self.block = AAUBlock(self.conv_ch * 2, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.block(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# -------------------- AAUNet 主体 --------------------
class AAUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, n_filters=None):
        super(AAUNet, self).__init__()
        if n_filters is None:
            # n_filters = [16, 32, 64, 128, 256]  # 可切换为更大通道
            n_filters = [32, 64, 128, 256, 512]
            # n_filters = [64, 128, 256, 512, 1024]

        self.inc = AAUBlock(n_channels, n_filters[0])
        self.down1 = Down(n_filters[0], n_filters[1])
        self.down2 = Down(n_filters[1], n_filters[2])
        self.down3 = Down(n_filters[2], n_filters[3])
        self.down4 = Down(n_filters[3], n_filters[4])

        self.up1 = Up(n_filters[4], n_filters[3], bilinear)
        self.up2 = Up(n_filters[3], n_filters[2], bilinear)
        self.up3 = Up(n_filters[2], n_filters[1], bilinear)
        self.up4 = Up(n_filters[1], n_filters[0], bilinear)

        self.outc = OutConv(n_filters[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# -------------------- 测试模型 --------------------
if __name__ == "__main__":
    model = AAUNet(n_channels=1, n_classes=1, bilinear=True)
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print("输入:", x.shape)
    print("输出:", y.shape)
