"""Model definitions for traffic sign classification (GTSRB).

This module provides a compact convolutional neural network tailored for the
German Traffic Sign Recognition Benchmark (GTSRB). The default model,
``GTSRB_CNN``, expects RGB images resized to 32x32 and produces class logits
for 43 categories.

Conventions
----------
- Input tensor shape: [N, 3, 32, 32] in RGB order.
- Inputs should be scaled to [0, 1] via ``ToTensor()`` and are typically
  normalized using dataset mean/std (see ``src.data_loader``).
- The forward method returns raw, unnormalized logits; apply ``softmax`` or
  ``argmax`` externally as needed.
"""

import torch
from torch import nn


class GTSRB_CNN(nn.Module):
    """A compact CNN for GTSRB classification.

    Architecture overview:
    - Feature extractor: 3 convolutional blocks (Conv -> ReLU -> BatchNorm -> MaxPool)
      with channel progression 3→32→64→128 and spatial downsampling by 2 per block
      (32x32 → 16x16 → 8x8 → 4x4).
    - Classifier: Flatten → Linear(2048→512) → ReLU → BatchNorm1d → Dropout(0.5)
      → Linear(512→num_classes).

    Notes:
    - Dropout is applied during training only.
    - Outputs are logits; combine with CrossEntropyLoss during training.

    Args:
        num_classes (int): Number of target classes. Defaults to 43 for GTSRB.
    """
    def __init__(self, num_classes=43):
        """Initialize layers for feature extraction and classification.

        Args:
            num_classes: Number of output classes for the dataset/task.
        """
        super(GTSRB_CNN, self).__init__()

        # --- Convolutional Feature Extractor ---
        # Progressively increases channel depth while reducing spatial size.
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Fully Connected Classifier ---
        # After 3 max-pooling layers of stride 2, a 32x32 image becomes 4x4.
        # So the flattened feature size is 128 * 4 * 4 = 2048.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Compute forward pass and return logits.

        Args:
            x (torch.Tensor): Input batch of shape ``[N, 3, 32, 32]`` with ``dtype``
                ``float32``. Values are expected in [0, 1], typically normalized
                per-channel using dataset statistics.

        Returns:
            torch.Tensor: Logits of shape ``[N, num_classes]``.

        Raises:
            RuntimeError: If the input has an unexpected shape or channel count.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # Simple self-test to validate tensor plumbing and parameter counts.
    print("--- Testing the GTSRB_CNN model architecture ---")

    # Create a dummy input tensor with the expected shape (batch_size, channels, height, width)
    dummy_input = torch.randn(16, 3, 32, 32)

    # Instantiate the model
    model = GTSRB_CNN(num_classes=43)

    # Perform a forward pass
    output = model(dummy_input)

    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Model output shape: {output.shape}")  # Expected: [16, 43]

    # Print the number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
