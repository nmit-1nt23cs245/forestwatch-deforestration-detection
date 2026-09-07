import torch
import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)


# ============================================================
# MODEL
# ============================================================

class ResNet50_21Channel(nn.Module):

    def __init__(
        self,
        num_classes=2,
        pretrained=True
    ):

        super().__init__()

        # ----------------------------------------------------
        # Load standard ResNet50
        # ----------------------------------------------------

        if pretrained:

            weights = (
                ResNet50_Weights.DEFAULT
            )

        else:

            weights = None

        self.model = resnet50(
            weights=weights
        )

        # ----------------------------------------------------
        # Replace first convolution
        #
        # Original ResNet50:
        #   3 input channels
        #
        # Experiment 3:
        #   21 input channels
        # ----------------------------------------------------

        old_conv = self.model.conv1

        self.model.conv1 = nn.Conv2d(
            in_channels=21,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # ----------------------------------------------------
        # Initialize the new 21-channel convolution
        #
        # First 3 channels:
        #   Original pretrained RGB weights
        #
        # Remaining 18 channels:
        #   Mean of pretrained RGB filters
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
                    18,
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

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x
    ):

        return self.model(x)


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 80)

    print(
        "FORESTWATCH - 21-CHANNEL RESNET50 TEST"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = ResNet50_21Channel(
        num_classes=2,
        pretrained=True
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Verify input channels
    # --------------------------------------------------------

    print(
        f"Input channels: "
        f"{model.model.conv1.in_channels}"
    )

    print(
        f"Output classes: "
        f"{model.model.fc.out_features}"
    )

    # --------------------------------------------------------
    # Dummy Experiment 3 input
    # --------------------------------------------------------

    x = torch.randn(
        2,
        21,
        224,
        224,
        device=device
    )

    print(
        f"\nInput shape : "
        f"{tuple(x.shape)}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            x
        )

    print(
        f"Output shape: "
        f"{tuple(output.shape)}"
    )

    print(
        "\nOutput:"
    )

    print(
        output
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert (
        model.model.conv1.in_channels
        == 21
    )

    assert (
        model.model.fc.out_features
        == 2
    )

    assert (
        output.shape
        == (2, 2)
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"\nParameters: "
        f"{total_parameters}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "✅ 21-CHANNEL RESNET50 "
        "FORWARD PASS SUCCESSFUL"
    )

    print(
        "=" * 80
    )