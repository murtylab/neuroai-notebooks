import torch
from dataclasses import dataclass

@dataclass
class RidgeResult:
    weight: torch.Tensor
    bias: float
    mean: torch.Tensor | None

    def __repr__(self):
        return (f"RidgeResult(weight_shape={self.weight.shape}, "
                f"bias={self.bias}, "
                f"mean_shape={self.mean.shape if self.mean is not None else None})")

def ridge_regression(X_train, Y_train, lam=1e-3, fit_intercept=True, device='cpu'):
    """
    Closed-form Ridge regression using PyTorch.

    Args:
        X_train (torch.Tensor): Training features, shape (N, D)
        Y_train (torch.Tensor): Training targets, shape (N,) or (N, M)
        lam (float): Regularization parameter (λ)
        fit_intercept (bool): Whether to include an intercept term
        device (str or torch.device): Device to use ('cpu' or 'cuda')

    Returns:
        RidgeResult:
            weight (torch.Tensor): Learned weights, shape (D,) or (D, M)
            bias (float or torch.Tensor): Intercept term(s)
            mean (torch.Tensor or None): Training feature mean, shape (1, D), or None if no intercept
    """
    device = torch.device(device)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)

    assert X_train.dim() == 2, f"X_train must be 2D (N, D), got {X_train.shape}"
    assert Y_train.dim() in (1, 2), f"Y_train must be 1D or 2D, got {Y_train.shape}"
    assert X_train.size(0) == Y_train.size(0), \
        f"Mismatched samples: X_train {X_train.size(0)}, Y_train {Y_train.size(0)}"

    N, D = X_train.shape

    X_np = X_train.cpu().numpy()
    Y_np = Y_train.cpu().numpy()

    model = Ridge(alpha=lam, fit_intercept=fit_intercept, solver='auto')
    model.fit(X_np, Y_np)

    w = torch.from_numpy(model.coef_).to(device).float()
    # model.coef_ shape: (D,) for 1D target, (M, D) for multi-target (scikit-learn returns (M, D))
    # We want (D,) or (D, M)
    if w.ndim == 2:
        w = w.T  # (M, D) -> (D, M)
    intercept = model.intercept_
    if isinstance(intercept, float) or isinstance(intercept, int):
        intercept = float(intercept)
    else:
        intercept = torch.tensor(intercept).to(device).float()  # shape (M,)

    X_mean = torch.from_numpy(X_np.mean(0, keepdims=True)).to(device).float() if fit_intercept else None

    return RidgeResult(weight=w, bias=intercept, mean=X_mean)

import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from .hook import ForwardHook
from sklearn.linear_model import Ridge

class RidgeModule(nn.Module):
    def __init__(
        self,
        ridge_result: RidgeResult,
        device: str = 'cpu',
    ):
        super().__init__()
        self.ridge_weight = ridge_result.weight.to(device=device)
        self.ridge_bias = ridge_result.bias
        self.feature_mean = ridge_result.mean.to(device=device) if ridge_result.mean is not None else None
        self.device = device

        assert isinstance(self.ridge_weight, torch.Tensor), \
            f"ridge_weight must be torch.Tensor, but got {type(self.ridge_weight)}"
        
    def forward(self, x):
        # Use training mean for centering (if available)
        assert x.ndim == 2, f"Expected a 2d tensor of shape (Batch, (channels x height x width)) but got: {x.shape}"
        if self.feature_mean is not None:
            x = x - self.feature_mean

        # Apply ridge regression weights
        
        logits = x @ self.ridge_weight + self.ridge_bias
        return logits

class RidgeModel(nn.Module):
    def __init__(
        self,
        backbone_model: nn.Module,
        transforms: callable,
        hook_layer_name: str,
        ridge_result: RidgeResult,
        device: str = 'cpu',
    ):
        super().__init__()

        assert isinstance(backbone_model, nn.Module), \
            f"backbone must be an nn.Module, but got {type(self.backbone_model)}"
        assert callable(transforms), \
            f"transforms must be callable, but got {type(transforms)}"
        assert isinstance(hook_layer_name, str), \
            f"hook_layer_name must be str, but got {type(self.hook_layer_name)}"
        assert isinstance(ridge_result, RidgeResult), \
            f"ridge_result must be RidgeResult, but got {type(ridge_result)}"


        self.backbone_model = backbone_model.to(device=device).eval()
        self.hook_layer_name = hook_layer_name
        self.ridge_module = RidgeModule(ridge_result, device=device)
        self.feature_mean = ridge_result.mean.to(device=device) if ridge_result.mean is not None else None
        self.device = device
        self.transforms = transforms
        
        self.hook = ForwardHook(
            model=self.backbone_model,
            hook_layer_name=self.hook_layer_name,
        )

    def forward_pass_on_ridge_params(self, features):
        return self.ridge_module(features)

    def evaluate(self, x_test, y_test) -> float:
        y_pred = self.forward_pass_on_ridge_params(
            features=x_test.to(self.device)
        )
        correlation_matrix = torch.corrcoef(torch.stack((y_test.to(self.device), y_pred)))
        correlation = correlation_matrix[0, 1].item()
        return correlation
    
    

    def run(self, images: list[Image.Image]) -> torch.Tensor:
        assert isinstance(images, list), f"images must be a list, got {type(images)}"
        assert all(isinstance(img, Image.Image) for img in images), \
            f"all elements in images must be PIL images"

        # ensure RGB
        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

        # Preprocess
        image_tensor = torch.stack([self.transforms(img) for img in images]).to(self.device)
        assert image_tensor.ndim == 4, f"Expected (B, C, H, W), got {image_tensor.shape}"
        assert image_tensor.size(1) == 3, f"Expected 3 channels, got {image_tensor.size(1)}"

        # Forward pass
        with torch.no_grad():
            _ = self.backbone_model(image_tensor)
            features = self.hook.output

        assert features.ndim == 4, f"Expected features (B, C, H, W), got {features.shape}"

        features = rearrange(features, 'b c h w -> b (c h w)')
        assert features.ndim == 2, f"Expected (B, D), got {features.shape}"

    
        logits = self.forward_pass_on_ridge_params(features)
        return logits.cpu().tolist()
    
    def forward(self, x):
        logits = self.backbone_model(x)
        features = self.hook.output
        features = rearrange(features, 'b c h w -> b (c h w)')
        return self.forward_pass_on_ridge_params(features=features)