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

# -------------------- 边界注意力模块 --------------------

# -------------------- CBAM 注意力模块 --------------------
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=11):
        super(CBAM, self).__init__()

        # 通道注意力部分
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 空间注意力部分
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_att(x)
        x_channel = x * channel_att

        # 空间注意力
        max_pool = torch.max(x_channel, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        x_spatial = x_channel * spatial_att

        return x_spatial


class SobelEdgeExtractor(nn.Module):
    def __init__(self, in_ch):
        super(SobelEdgeExtractor, self).__init__()

        self.sobel_x = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False, groups=in_ch)
        self.sobel_y = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False, groups=in_ch)

        # 初始化Sobel核并克隆为独立内存
        sobel_kernel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)

        weight_x = sobel_kernel_x.view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1).clone()
        weight_y = sobel_kernel_y.view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1).clone()

        self.sobel_x.weight = nn.Parameter(weight_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(weight_y, requires_grad=False)

        self.cbam = CBAM(in_ch, reduction_ratio=8)

    def forward(self, x):
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        edge = self.cbam(edge)
        return edge

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
    def __init__(self, n_channels, n_classes, bilinear=True, n_filters=None):
        super(AAUNet, self).__init__()
        if n_filters is None:
            n_filters = [32, 64, 128, 256, 512]
            # n_filters = [16, 32, 64, 128, 256]
            # n_filters = [24, 48, 96, 192, 384]  # 保持 ×2 增长，但整体缩小 25%

            # 每层增长约 1.5x

        self.edge_extractor1 = SobelEdgeExtractor(n_filters[0])
        self.edge_extractor2 = SobelEdgeExtractor(n_filters[1])

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
        e1 = self.edge_extractor1(x1)

        x2 = self.down1(x1)
        e2 = self.edge_extractor2(x2)

        # 拼接边界图，生成注意力图
        edge_attention = torch.sigmoid(torch.mean(torch.cat([e1, F.interpolate(e2, size=e1.shape[2:], mode='bilinear', align_corners=False)], dim=1), dim=1, keepdim=True))

        # 编码
        # x2 = x2 * F.interpolate(edge_attention, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2 * F.interpolate(edge_attention, size=x2.shape[2:], mode='bilinear', align_corners=False))
        x = self.up4(x, x1 * F.interpolate(edge_attention, size=x1.shape[2:], mode='bilinear', align_corners=False))
        return self.outc(x)

# -------------------- 测试模型 --------------------
if __name__ == "__main__":
    from thop import profile
    import torch
    import torch.nn as nn

    # 你的模型
    model = AAUNet(n_channels=1, n_classes=1, bilinear=True)
    model.eval()

    # 输入尺寸 (batch_size=1 通常用于FLOPs计算)
    input_tensor = torch.randn(1, 1, 256, 256)  # 例如 1 通道 256×256 图像输入

    # 计算 FLOPs 和 参数量
    flops, params = profile(model, inputs=(input_tensor,))

    # 输出结果（单位转换为 M、G）
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")
    #
    # model = AAUNet(n_channels=1, n_classes=1, bilinear=True)
    # # x = torch.randn(1, 1, 256, 256)
    # # y = model(x)
    # # print("输入:", x.shape)
    # # print("输出:", y.shape)
    #
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}")

