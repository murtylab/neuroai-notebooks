import numpy as np
from torchtyping import TensorType
import torch
## import squareform and pdist from scipy
from scipy.spatial.distance import squareform, pdist

@torch.no_grad()
def rdm_from_predictions(fmri_predictions: TensorType["samples", "voxels"], mode = "scipy", device = "cpu") -> np.ndarray:
    assert fmri_predictions.ndim == 2, f'Expected fmri_predictions to be 2D (samples, voxels), but got {fmri_predictions.shape}'
    num_voxels = fmri_predictions.shape[1]
    if mode == "scipy":
        fmri_predictions = fmri_predictions.cpu().numpy()
        rdm = squareform(pdist(fmri_predictions, metric="euclidean"))
    else:
        fmri_predictions = fmri_predictions.to(device)
        rdm = torch.cdist(fmri_predictions, fmri_predictions, p=2)

    rdm = rdm.cpu().numpy() if isinstance(rdm, torch.Tensor) else rdm
    assert rdm.shape[0] == rdm.shape[1], f"RDM is not square: {rdm.shape}"

    ## Normalize the RDM
    rdm = rdm / num_voxels
    return rdm