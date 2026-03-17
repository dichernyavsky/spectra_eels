import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct1D(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 32,
        activation: str = "relu",
    ):
        super().__init__()
        padding = "same"
        self.conv = nn.Conv1d(
            c_in,
            c_out,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.bn = nn.BatchNorm1d(c_out, eps=1e-3, momentum=0.01)

        if activation == "relu":
            self.act = nn.ReLU(inplace=False)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class PaperUNet1D(nn.Module):
    """
    PyTorch reproduction of the paper's TF/Keras U-Net-like classifier
    based on model.summary():

    Input:
        x    : [B, 1, 3072]
        mask : [B, 1, 3072]  (ignored by the network body, accepted for pipeline compatibility)

    Output:
        dict with:
            logits : [B, 80]
            probs  : [B, 80]
    """

    def __init__(
        self,
        num_classes: int = 80,
        activation: str = "relu",
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        # Encoder
        self.enc1 = ConvBNAct1D(1, 16, kernel_size=32, activation=activation)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc2 = ConvBNAct1D(16, 32, kernel_size=32, activation=activation)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc3 = ConvBNAct1D(32, 64, kernel_size=32, activation=activation)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc4 = ConvBNAct1D(64, 128, kernel_size=32, activation=activation)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBNAct1D(128, 256, kernel_size=32, activation=activation)

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec4a = ConvBNAct1D(256, 128, kernel_size=32, activation=activation)
        self.dec4b = ConvBNAct1D(128 + 128, 128, kernel_size=32, activation=activation)

        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3a = ConvBNAct1D(128, 64, kernel_size=32, activation=activation)
        self.dec3b = ConvBNAct1D(64 + 64, 64, kernel_size=32, activation=activation)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2a = ConvBNAct1D(64, 32, kernel_size=32, activation=activation)
        self.dec2b = ConvBNAct1D(32 + 32, 32, kernel_size=32, activation=activation)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1a = ConvBNAct1D(32, 16, kernel_size=32, activation=activation)
        self.dec1b = ConvBNAct1D(16 + 16, 16, kernel_size=32, activation=activation)

        # Final conv to 1 channel (Keras uses activation='relu' here)
        self.final_conv = nn.Conv1d(16 + 16, 1, kernel_size=1, padding=0, bias=True)
        self.final_relu = nn.ReLU(inplace=False)

        # Dense head
        self.fc1 = nn.Linear(3072, 3072)
        self.bn1 = nn.BatchNorm1d(3072, eps=1e-3, momentum=0.01)
        self.drop1 = nn.Dropout(dropout_p)

        self.fc2 = nn.Linear(3072, 1536)
        self.bn2 = nn.BatchNorm1d(1536, eps=1e-3, momentum=0.01)
        self.drop2 = nn.Dropout(dropout_p)

        self.fc3 = nn.Linear(1536, 768)
        self.bn3 = nn.BatchNorm1d(768, eps=1e-3, momentum=0.01)
        self.drop3 = nn.Dropout(dropout_p)

        self.fc4 = nn.Linear(768, 384)
        self.bn4 = nn.BatchNorm1d(384, eps=1e-3, momentum=0.01)

        self.fc_out = nn.Linear(384, num_classes)

        if activation == "relu":
            self.head_act = nn.ReLU(inplace=False)
        elif activation == "gelu":
            self.head_act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.apply(self._init_weights)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"EELSModel (paper-like 1D U-Net) initialized with {num_params} trainable parameters.")

    def _init_weights(self, module: nn.Module) -> None:
        """Keras he_normal–like init: Conv1d and Linear get kaiming_normal, bias zeros; BatchNorm untouched."""
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def init_bias_from_prevalence(self, prevalence: torch.Tensor):
        """
        prevalence: [K]
        Initialize final output bias so that sigmoid(bias) ~= prevalence.
        """
        p = prevalence.clamp(1e-4, 1.0 - 1e-4)
        bias = torch.log(p / (1.0 - p))
        with torch.no_grad():
            self.fc_out.bias.copy_(bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        # x: [B, 1, 3072]
        # mask ignored intentionally to stay close to the paper model

        # Encoder
        s1 = self.enc1(x)          # [B, 16, 3072]
        x1 = self.pool1(s1)        # [B, 16, 1536]

        s2 = self.enc2(x1)         # [B, 32, 1536]
        x2 = self.pool2(s2)        # [B, 32, 768]

        s3 = self.enc3(x2)         # [B, 64, 768]
        x3 = self.pool3(s3)        # [B, 64, 384]

        s4 = self.enc4(x3)         # [B, 128, 384]
        x4 = self.pool4(s4)        # [B, 128, 192]

        # Bottleneck
        b = self.bottleneck(x4)    # [B, 256, 192]

        # Decoder
        u4 = self.up4(b)           # [B, 256, 384]
        u4 = self.dec4a(u4)        # [B, 128, 384]
        u4 = torch.cat([u4, s4], dim=1)  # [B, 256, 384]
        u4 = self.dec4b(u4)        # [B, 128, 384]

        u3 = self.up3(u4)          # [B, 128, 768]
        u3 = self.dec3a(u3)        # [B, 64, 768]
        u3 = torch.cat([u3, s3], dim=1)  # [B, 128, 768]
        u3 = self.dec3b(u3)        # [B, 64, 768]

        u2 = self.up2(u3)          # [B, 64, 1536]
        u2 = self.dec2a(u2)        # [B, 32, 1536]
        u2 = torch.cat([u2, s2], dim=1)  # [B, 64, 1536]
        u2 = self.dec2b(u2)        # [B, 32, 1536]

        u1 = self.up1(u2)          # [B, 32, 3072]
        u1 = self.dec1a(u1)        # [B, 16, 3072]
        u1 = torch.cat([s1, u1], dim=1)  # [B, 32, 3072]

        # Final conv
        z = self.final_conv(u1)    # [B, 1, 3072]
        z = self.final_relu(z)
        z = z.squeeze(1)           # [B, 3072]

        # Dense head
        z = self.fc1(z)
        z = self.bn1(z)
        z = self.head_act(z)
        z = self.drop1(z)

        z = self.fc2(z)
        z = self.bn2(z)
        z = self.head_act(z)
        z = self.drop2(z)

        z = self.fc3(z)
        z = self.bn3(z)
        z = self.head_act(z)
        z = self.drop3(z)

        z = self.fc4(z)
        z = self.bn4(z)
        z = self.head_act(z)

        logits = self.fc_out(z)    # [B, 80]
        probs = torch.sigmoid(logits)

        return {
            "logits": logits,
            "probs": probs,
        }


# Keep the expected name for train.py
EELSModel = PaperUNet1D