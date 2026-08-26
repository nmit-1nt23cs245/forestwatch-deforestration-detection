import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50_14Channel(nn.Module):

    def __init__(self, num_classes=2, pretrained=True):

        super().__init__()

        # ----------------------------------------------------
        # Load standard ResNet50
        # ----------------------------------------------------

        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None

        self.model = resnet50(
            weights=weights
        )

        # ----------------------------------------------------
        # Replace first convolution:
        #
        # Original:
        #   3 input channels
        #
        # ForestWatch:
        #   14 input channels
        # ----------------------------------------------------

        old_conv = self.model.conv1

        self.model.conv1 = nn.Conv2d(
            in_channels=14,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # ----------------------------------------------------
        # Initialize the new 14-channel convolution
        #
        # The first 3 channels retain the pretrained weights.
        # The remaining channels use the mean of the original
        # RGB filters.
        # ----------------------------------------------------

        with torch.no_grad():

            self.model.conv1.weight[:, :3] = (
                old_conv.weight
            )

            mean_weights = (
                old_conv.weight.mean(
                    dim=1,
                    keepdim=True
                )
            )

            self.model.conv1.weight[:, 3:] = (
                mean_weights.repeat(
                    1,
                    11,
                    1,
                    1
                )
            )

        # ----------------------------------------------------
        # Replace classification head
        # ----------------------------------------------------

        in_features = (
            self.model.fc.in_features
        )

        self.model.fc = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):

        return self.model(x)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("FORESTWATCH - 14-CHANNEL RESNET50 TEST")
    print("=" * 70)

    print(
        f"\nDevice: {device}"
    )

    model = ResNet50_14Channel(
        num_classes=2,
        pretrained=True
    )

    model = model.to(device)

    print(
        f"Input channels: "
        f"{model.model.conv1.in_channels}"
    )

    print(
        f"Output classes: "
        f"{model.model.fc.out_features}"
    )

    # --------------------------------------------------------
    # Dummy ForestWatch input
    # --------------------------------------------------------

    x = torch.randn(
        2,
        14,
        224,
        224,
        device=device
    )

    print(
        f"\nInput shape : "
        f"{tuple(x.shape)}"
    )

    with torch.no_grad():

        output = model(x)

    print(
        f"Output shape: "
        f"{tuple(output.shape)}"
    )

    print(
        "\nOutput:"
    )

    print(output)

    print(
        "\n✅ 14-channel ResNet50 forward pass successful."
    )

    print("=" * 70)