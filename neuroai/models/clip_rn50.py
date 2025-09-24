import os
import torchvision.transforms as transforms
import torch.nn as nn
import clip

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

model_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    ]
)


class CLIPRN50(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        download_root: str = "./checkpoints/clip_rn50_checkpoints",
    ):
        super().__init__()
        os.makedirs(download_root, exist_ok=True)
        

        clip_model, preprocess = clip.load(
            "RN50", device="cpu", download_root=download_root
        )
        self.model = clip_model.visual.float()

        if pretrained == False:
            for layer in self.model.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

        self.transforms = model_transforms

    @classmethod
    def get_transforms(cls):
        return model_transforms
    
    def forward(self, x):
        return self.model(x)
    