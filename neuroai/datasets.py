import os
import torch
import numpy as np
from einops import rearrange
from PIL import Image
import numpy as np
from .utils.seed import seed_everything


# https://drive.google.com/file/d/17iuOTy_I-LhORK24C_KXw2OA43Q4acxW/view?usp=share_link
def download_and_extract_nsd(zip_filename: str, output_folder: str = "data_nsd"):
    ## TODO: remove __MACOSX folder from the zip
    gdrive_id = "17iuOTy_I-LhORK24C_KXw2OA43Q4acxW"
    os.system(f"gdown --id {gdrive_id} -O {zip_filename}")
    unzip_command = f"unzip {zip_filename} -d {os.path.join(os.path.dirname(zip_filename), output_folder)}"
    os.system(
        unzip_command
    )
    ## find the folder just above
    data_nsd_folder = os.path.join(
        output_folder, "data/nsd/"
    ) 
    os.system(
        f"mv {data_nsd_folder} {os.path.join(output_folder, 'nsd')}"
    )
    os.system(f"rm -rf {data_nsd_folder}")
    print(f"Extracted NSD dataset to {output_folder}")



valid_subject_ids = ["s1", "s2", "s5", "s7"]
valid_regions = [
    "eba",
    "fba",
    "ffa",
    "ofa",
    "opa",
    "ppa",
    "rsc",
    "vwfa",
]


def validate_dataset_files(folder: str, loud=False):
    for subject_id in valid_subject_ids:
        for region in valid_regions:
            filename = os.path.join(folder, f"{subject_id}_{region.upper()}_t7.pt")
            assert os.path.exists(
                filename
            ), f"Expected {filename} to exist in {folder} but it was not found :("
    if loud:
        print("\033[92mAll files are present in the dataset folder :)\033[0m")


class NSDStimuli:
    def __init__(self, folder: str):
        validate_dataset_files(folder=folder)
        self.folder = folder
        self.data = np.load(os.path.join(folder, "nsd_stimuli1000.npy"))
        assert self.data.ndim == 4
        self.data = rearrange(self.data, "b h w c -> b c h w")

    def __getitem__(self, index: int):
        assert index in range(
            len(self.data)
        ), f"Expected index to be in range {len(self.data)} but got {index}"
        return self.data[index]

    def __len__(self):
        return len(self.data)


class NSDSingleSubjectSingleRegion:
    def __init__(self, folder: str, subject_id: str = "s1", region: str = "EBA"):
        validate_dataset_files(folder=folder)
        self.folder = folder
        self.subject_id = subject_id
        self.region = region
        self.data = torch.load(
            os.path.join(folder, f"{subject_id}_{region.upper()}_t7.pt"), weights_only=True
        )
        self.stimuli = NSDStimuli(folder=folder)
    def __len__(self):
        return len(self.data)

    def validate_args(self, subject_id: str, region: str):

        assert (
            subject_id in valid_subject_ids
        ), f"Expected subject_id to be one of {valid_subject_ids} but got {subject_id}"
        assert (
            region in valid_regions
        ), f"Expected region to be one of {valid_regions} but got {region}"

    def __getitem__(self, index: int):

        assert index in range(
            len(self.data)
        ), f"Expected index to be in range {len(self.data)} but got {index}"
        return self.data[index]


class NSDAllSubjectSingleRegion:
    def __init__(
        self,
        folder: str,
        region: str = "EBA",
        flatten=False,
        transforms: callable = None,
        subset="train",
        train_test_split=0.8,
    ):
        self.datasets = {}
        seed_everything(0)

        self.stimuli = NSDStimuli(folder=folder)
        self.flatten = flatten
        self.transforms = transforms

        # Determine split indices
        total_len = len(self.stimuli)
        indices = np.arange(total_len)
        np.random.shuffle(indices)
        split_idx = int(total_len * train_test_split)
        if subset == "train":
            self.selected_indices = indices[:split_idx]
        elif subset == "test":
            self.selected_indices = indices[split_idx:]
        else:
            raise ValueError(f"Unknown subset: {subset}")

        for subject_id in valid_subject_ids:
            self.datasets[subject_id] = NSDSingleSubjectSingleRegion(
                folder=folder, subject_id=subject_id, region=region
            )
            assert self.stimuli.data.shape[0] == self.datasets[subject_id].data.shape[0]

    def __getitem__(self, idx: int):
        assert idx in range(
            len(self.selected_indices)
        ), f"Expected idx to be in range {len(self.selected_indices)} but got {idx}"

        index = self.selected_indices[idx]

        if not self.flatten:
            responses = {
                subject_id: dataset[index]
                for subject_id, dataset in self.datasets.items()
            }
        else:
            responses = torch.cat(
                [dataset[index] for dataset in self.datasets.values()], dim=0
            )

        if self.transforms is not None:
            image_hwc = rearrange(self.stimuli[index], "c h w -> h w c")
            image_tensor = self.transforms(Image.fromarray(image_hwc))
        else:
            image_tensor = self.stimuli[index]

        return {"image": image_tensor, "fmri_response": responses}

    def __len__(self):
        return len(self.selected_indices)