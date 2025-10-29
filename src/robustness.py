"""Robustness and corruption transforms for evaluation.

This module defines lightweight, composable image corruptions used to probe model
robustness under controlled severity levels. It provides:

- Custom transforms: ``SaltAndPepperNoise``, ``GaussianNoise``, ``RandomSquareOcclusion``
- A factory function ``get_robustness_transform`` to build a torchvision ``Compose``
    for a selected corruption and severity.

Design notes
------------
- All transforms operate on tensor images (C x H x W), typically produced by
    ``torchvision.transforms.ToTensor`` with values in [0, 1].
- We intentionally place corruptions after ``ToTensor`` and before any normalization
    so that severity factors are interpretable. If you normalize inputs, insert your
    normalization transform after the returned corruption pipeline.
- Severity levels are mapped discretely (1..5) to pre-tuned parameters per corruption
    type. You can adjust these mappings for your needs.
"""

import random
import torch
import torchvision.transforms as T
import numpy as np

# --- Helper Classes for Custom Transformations ---

class SaltAndPepperNoise:
    """Apply salt-and-pepper impulse noise to a tensor image (in-place).

    This transform randomly sets a subset of pixels to 0.0 (pepper) or 1.0 (salt).
    It modifies the provided tensor directly for efficiency.

    Args:
        amount (float): Approximate fraction of image pixels to flip to salt and to pepper
            respectively (applied separately). For example, ``amount=0.05`` will select ~5%
            of pixels for salt and ~5% for pepper. Expected range: [0, 1].

    Notes:
        - Input is expected to be a float tensor of shape C x H x W with values in [0, 1].
        - This operation is performed in-place. If you need to retain the original tensor,
          pass a clone of the image to the transform.
    """
    def __init__(self, amount=0.05):
        self.amount = amount

    def __call__(self, img):
        # img is a Tensor (C x H x W)
        c, h, w = img.shape
        num_salt = int(np.ceil(self.amount * h * w))
        num_pepper = int(np.ceil(self.amount * h * w))

        if num_salt > 0:
            ys = torch.randint(0, h, (num_salt,))
            xs = torch.randint(0, w, (num_salt,))
            img[:, ys, xs] = 1.0  # apply to all channels at selected pixels

        if num_pepper > 0:
            ys = torch.randint(0, h, (num_pepper,))
            xs = torch.randint(0, w, (num_pepper,))
            img[:, ys, xs] = 0.0  # apply to all channels at selected pixels

        return img


class GaussianNoise:
    """Additive Gaussian noise for tensor images.

    Args:
        mean (float): Mean of the Gaussian distribution to add (default: 0.0).
        std (float): Standard deviation of the Gaussian distribution (default: 0.1).

    Returns:
        torch.Tensor: Noisy tensor image of the same shape as input. The operation returns
        a new tensor and does not modify the input in-place.

    Notes:
        - No clamping is performed; downstream pipelines may clip or normalize as needed.
        - Input is expected to be a float tensor of shape C x H x W.
    """
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        # Preserve device/dtype of the input tensor
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class RandomSquareOcclusion:
    """Occlude a square patch at a random location using a fixed pixel size.

    This simple proxy for occlusion makes the severity directly control the
    square side length in pixels, enabling predictable scaling of the occluded
    area. Given a fixed input size (e.g., 32x32), using sizes like 4, 6, 8, 10,
    and 12 provides a visibly increasing obstruction.

    Args:
        size (int): Side length of the square patch in pixels (e.g., 4, 6, 8...).
        value (float): Fill value for the occluded region (0.0 = black). Defaults to 0.0.
        per_channel (bool): If True, occludes each channel independently; otherwise, the
            square is applied across all channels (typical for RGB occlusion).

    Notes:
        - Operates in-place on the provided tensor (C x H x W). Clone beforehand if needed.
        - The patch location is sampled uniformly so that the entire square fits in-frame.
    """

    def __init__(self, size: int, value: float = 0.0, per_channel: bool = False):
        self.size = int(size)
        self.value = float(value)
        self.per_channel = per_channel

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img expected shape: C x H x W
        c, h, w = img.shape
        if self.size <= 0:
            return img

        sz = int(min(self.size, h, w))
        # Random top-left so patch fits in the image
        top = torch.randint(0, h - sz + 1, (1,)).item()
        left = torch.randint(0, w - sz + 1, (1,)).item()

        if self.per_channel:
            # Occlude per-channel independently
            for ch in range(c):
                img[ch, top:top + sz, left:left + sz] = self.value
        else:
            # Occlude all channels together
            img[:, top:top + sz, left:left + sz] = self.value
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, value={self.value}, per_channel={self.per_channel})"


# --- Main Factory Function ---

def get_robustness_transform(corruption_type, severity=1, random_state=42):
    """Build a torchvision transform pipeline for a given corruption and severity.

    The returned pipeline always includes ``Resize((32, 32))`` and ``ToTensor`` and,
    when requested, appends a corruption transform parameterized by the severity level.
    Severity levels are discrete and mapped to corruption parameters defined below.

    Args:
        corruption_type (str): One of {'gaussian_noise', 'salt_pepper', 'brightness',
            'contrast', 'rotation', 'occlusion'}.
        severity (int): Discrete severity level in {1, 2, 3, 4, 5}.
        random_state (int): Seed applied to ``torch``, ``numpy``, and Python's ``random``
            for deterministic behavior across runs.

    Returns:
        torchvision.transforms.Compose: A composed transform pipeline.

    Notes:
        - Normalization (if any) should be applied after this pipeline.
        - Invalid ``corruption_type`` will raise a ``KeyError``; out-of-range ``severity``
          will likely raise an ``IndexError`` when indexing the severity mapping.
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)

    # Base transforms that are always applied. ``ToTensor`` converts to CxHxW float in [0, 1].
    base_transforms = [T.Resize((32, 32)), T.ToTensor()]

    # Severity levels (example values; tune as desired for your evaluation needs)
    severities = {
        'gaussian_noise': [0.05, 0.1, 0.15, 0.2, 0.25],
        'salt_pepper':    [0.01, 0.02, 0.03, 0.04, 0.05],
        'brightness':     [0.1, 0.3, 0.5, 0.7, 0.9], # Factor to adjust brightness
        'contrast':       [0.1, 0.3, 0.5, 0.7, 0.9], # Factor to adjust contrast
        'rotation':       [5, 10, 15, 20, 25],      # Max degrees
        'occlusion':      [4, 6, 8, 10, 12]        # Square patch side in pixels (for 32x32 input)
    }

    s = severities[corruption_type][severity - 1]

    corruption_transform = None
    if corruption_type == 'gaussian_noise':
        corruption_transform = GaussianNoise(std=s)
    elif corruption_type == 'salt_pepper':
        corruption_transform = SaltAndPepperNoise(amount=s)
    elif corruption_type == 'brightness':
        # Adjusts brightness. factor=1 gives original. 0 is black. 2 is twice as bright.
        corruption_transform = T.ColorJitter(brightness=s)
    elif corruption_type == 'contrast':
        corruption_transform = T.ColorJitter(contrast=s)
    elif corruption_type == 'rotation':
        corruption_transform = T.RandomRotation(degrees=(-s, s))
    elif corruption_type == 'occlusion':
        # Use a custom square occlusion whose size is driven by severity.
        # For a 32x32 input, sizes like 4,6,8,10,12 correspond to progressively
        # larger occluded areas, making severity interpretable and consistent.
        corruption_transform = RandomSquareOcclusion(size=s, value=0.0, per_channel=False)

    if corruption_transform:
        # Append corruption after tensor conversion and prior to any normalization.
        return T.Compose(base_transforms + [corruption_transform])
    return T.Compose(base_transforms)
