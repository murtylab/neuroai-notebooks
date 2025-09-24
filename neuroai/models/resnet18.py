import os
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet18

IMAGENET_STANDARD_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STANDARD_STD = (0.229, 0.224, 0.225)

model_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD),
    ]
)

class ResNet18(nn.Module):
    def __init__(
        self, pretrained: bool = False, download_root="./checkpoints/resnet18_checkpoints"
    ):
        super().__init__()
        os.makedirs(download_root, exist_ok=True)

        if not pretrained:
            self.model = resnet18(weights=None).to(device="cpu")
        else:
            self.model = resnet18(weights="DEFAULT").to(device="cpu")

        self.transforms = model_transforms
        self.model.eval()

    @classmethod
    def get_transforms(cls):
        return model_transforms

    def forward(self, x):
        return self.model(x)
