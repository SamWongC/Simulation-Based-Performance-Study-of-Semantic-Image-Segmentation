"""LightUNet, the compact U-Net-style segmentation network of this study, and a
helper that draws its architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnRelu(nn.Sequential):
    """Convolution (3×3 by default) → BatchNorm → ReLU; each LightUNet stage stacks two."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


class LightUNet(nn.Module):
    """Small U-Net-style encoder-decoder with skip connections.

    Four encoder stages (32 to 256 channels, each followed by 2× max-pooling), a
    512-channel bottleneck, and four decoder stages that upsample (bilinear) and
    concatenate the matching encoder features. Maps (B, 3, H, W) images to
    (B, num_classes, H, W) logits.
    """
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
        """Return per-pixel class logits for a batch of normalised images."""
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
    

def plot_architecture():
    """Draw the LightUNet architecture as 3-D blocks.
    """
    import math
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    STAGES = {
        "Input":      (3,   1.0),
        "enc1":       (32,  1.0),
        "enc2":       (64,  0.5),
        "enc3":       (128, 0.25),
        "enc4":       (256, 0.125),
        "bottleneck": (512, 0.0625),
        "dec4":       (128, 0.125),
        "dec3":       (64,  0.25),
        "dec2":       (32,  0.5),
        "dec1":       (32,  1.0),
        "head":       (8,   1.0),
    }
    ORDER = list(STAGES.keys())
    COLOR = {"Input": "#e8e8e8", "enc1": "#9fc5e8", "enc2": "#9fc5e8", "enc3": "#9fc5e8", "enc4": "#9fc5e8",
             "bottleneck": "#f9cb9c", "dec4": "#b6d7a8", "dec3": "#b6d7a8", "dec2": "#b6d7a8", "dec1": "#b6d7a8",
             "head": "#e8e8e8"}

    def shade(hexcolor, factor):
        h = hexcolor.lstrip('#')
        r, g, b = [int(h[i:i+2], 16) for i in (0, 2, 4)]
        r, g, b = [min(255, max(0, int(c * factor))) for c in (r, g, b)]
        return f'#{r:02x}{g:02x}{b:02x}'

    def channel_to_depth(ch, min_ch=3, max_ch=512, min_d=0.10, max_d=0.85):
        t = (math.log(ch) - math.log(min_ch)) / (math.log(max_ch) - math.log(min_ch))
        return min_d + t * (max_d - min_d)

    MAX_H, FRONT_W, GAP = 2.2, 0.55, 0.75
    SKEW = (0.32, 0.22)  # (dx, dy) direction per unit depth, gives the isometric look

    fig, ax = plt.subplots(figsize=(16, 6))

    x = 0.0
    pos = {}
    for name in ORDER:
        ch, hfrac = STAGES[name]
        h = MAX_H * hfrac
        y = (MAX_H - h) / 2
        depth = channel_to_depth(ch)
        dx, dy = SKEW[0] * depth, SKEW[1] * depth
        color = COLOR[name]

        ax.fill([x, x+FRONT_W, x+FRONT_W, x], [y, y, y+h, y+h],
                color=color, edgecolor="black", linewidth=1.1, zorder=3)
        ax.fill([x, x+FRONT_W, x+FRONT_W+dx, x+dx], [y+h, y+h, y+h+dy, y+h+dy],
                color=shade(color, 1.12), edgecolor="black", linewidth=1.0, zorder=3)
        ax.fill([x+FRONT_W, x+FRONT_W, x+FRONT_W+dx, x+FRONT_W+dx], [y, y+h, y+h+dy, y+dy],
                color=shade(color, 0.78), edgecolor="black", linewidth=1.0, zorder=3)

        pos[name] = dict(x=x, y=y, h=h, dx=dx, dy=dy)
        ax.text(x + FRONT_W/2 + dx/2, -0.35, name, ha="center", va="top", fontsize=10.5, fontweight="bold")
        ax.text(x + FRONT_W/2 + dx/2, -0.62, f"{ch} ch", ha="center", va="top", fontsize=9, color="#444444")

        x += FRONT_W + dx + GAP

    def arrow(p0, p1, color="black", lw=1.3, rad=0.0):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                                      linewidth=lw, color=color, zorder=2,
                                      connectionstyle=f"arc3,rad={rad}"))

    POOL_AFTER = {"enc1", "enc2", "enc3", "enc4"}          # pool follows these
    UPSAMPLE_BEFORE = {"dec3", "dec2", "dec1"}              # up() feeds into these
    for a, b in zip(ORDER, ORDER[1:]):
        pa, pb = pos[a], pos[b]
        y_mid_a = pa["y"] + pa["h"] / 2 + pa["dy"] / 2
        y_mid_b = pb["y"] + pb["h"] / 2 + pb["dy"] / 2
        x0 = pa["x"] + FRONT_W + pa["dx"]
        x1 = pb["x"]
        arrow((x0, y_mid_a), (x1, y_mid_b))
        label = None
        if a in POOL_AFTER and b != "bottleneck":
            label = "pool 2×"
        elif a == "enc4" and b == "bottleneck":
            label = "pool 2×"
        elif a == "bottleneck" or b in UPSAMPLE_BEFORE or a in {"dec4", "dec3", "dec2"}:
            if b != "head":
                label = "↑ upsample"
        if label:
            ax.text((x0 + x1) / 2, max(y_mid_a, y_mid_b) + 0.34, label,
                     fontsize=8.5, color="#333333", ha="center")

    top_clear = max(p["y"] + p["h"] + p["dy"] for p in pos.values()) + 0.25
    skip_pairs = [("enc4", "dec4"), ("enc3", "dec3"), ("enc2", "dec2"), ("enc1", "dec1")]
    for i, (e, d) in enumerate(skip_pairs):
        pe, pd = pos[e], pos[d]
        x_e = pe["x"] + FRONT_W / 2 + pe["dx"] / 2
        x_d = pd["x"] + FRONT_W / 2 + pd["dx"] / 2
        y_e0 = pe["y"] + pe["h"] + pe["dy"] + 0.1
        y_d0 = pd["y"] + pd["h"] + pd["dy"] + 0.1
        y_bus = top_clear + i * 0.35  # innermost pair (enc4-dec4) rides the lowest bus
        ax.plot([x_e, x_e], [y_e0, y_bus], color="#777777", lw=1.1, zorder=2)
        ax.plot([x_e, x_d], [y_bus, y_bus], color="#777777", lw=1.1, zorder=2)
        arrow((x_d, y_bus), (x_d, y_d0), color="#777777", lw=1.1)
        ax.text(x_d - 0.1, y_bus + 0.08, "concat", fontsize=9, color="#666666",
                 style="italic", ha="right", va="bottom")

    legend_handles = [
        mpatches.Patch(facecolor="#9fc5e8", edgecolor="black", label="Encoder block (Conv+BN+ReLU x2)"),
        mpatches.Patch(facecolor="#f9cb9c", edgecolor="black", label="Bottleneck"),
        mpatches.Patch(facecolor="#b6d7a8", edgecolor="black", label="Decoder block (Conv+BN+ReLU x2)"),
        mpatches.Patch(facecolor="#e8e8e8", edgecolor="black", label="Input / Output (1x1 conv)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=4, frameon=False, fontsize=10)

    ax.set_xlim(-0.3, x + 0.3)
    ax.set_ylim(-1.7, MAX_H + 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("LightUNet: encoder-decoder with skip connections\n"
                  "(block height = spatial resolution, block depth = channel count -- both from model.py)",
                  fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LightUNet().to(device)
    x = torch.randn(2,3,256,512, device=device)
    out = model(x)
    params= sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Output={tuple(out.shape)}  params={params:.1f}M")
