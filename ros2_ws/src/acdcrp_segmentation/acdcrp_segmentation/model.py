import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


class LightUNet(nn.Module):
    # Small encoder-decoder with skip connections.

    # Input (B, 3, H, W)
    # Output (B, num_classes, H, W)

    def __init__(self, num_classes: int = 8):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(ConvBnRelu(3, 32), ConvBnRelu(32,32))
        self.enc2 = nn.Sequential(ConvBnRelu(32, 64), ConvBnRelu(64,64))
        self.enc3 = nn.Sequential(ConvBnRelu(64, 128), ConvBnRelu(128,128))
        self.enc4 = nn.Sequential(ConvBnRelu(128, 256), ConvBnRelu(256,256))

        self.pool = nn.MaxPool2d(2,2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBnRelu(256, 512),
            ConvBnRelu(512, 512),
        )

        # Decoder

        self.dec4 = nn.Sequential(ConvBnRelu(512+256, 256), ConvBnRelu(256, 128))
        self.dec3 = nn.Sequential(ConvBnRelu(128+128, 128), ConvBnRelu(128, 64))
        self.dec2 = nn.Sequential(ConvBnRelu(64+64, 64), ConvBnRelu(64, 32))
        self.dec1 = nn.Sequential(ConvBnRelu(32+32, 32), ConvBnRelu(32, 32))

        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        b = self.bottleneck(self.pool(s4))

        def up(feat, skip):
            feat = F.interpolate(feat, size=skip.shape[-2:],
                                 mode="bilinear", align_corners=False)
            return torch.cat([feat, skip], dim=1)

        d4 = self.dec4(up(b, s4))
        d3 = self.dec3(up(d4, s3))
        d2 = self.dec2(up(d3, s2))
        d1 = self.dec1(up(d2, s1))

        return self.head(d1) # (B, num_classes, H, W)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LightUNet().to(device)
    x = torch.randn(2,3,256,512, device=device)
    out = model(x)
    params= sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Output={tuple(out.shape)}  params={params:.1f}M")
