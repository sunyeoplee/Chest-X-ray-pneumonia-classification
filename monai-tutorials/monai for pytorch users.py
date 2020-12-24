# title : MONAI for PyTorch users
# author: Sun Yeop Lee

import numpy as np

import torch 

import warnings
warnings.filterwarnings('ignore') # remove some scikit-image warnings

import monai 
    # pip install monai[all] to install all optional dependencies which include nibabel, skimage, pillow, tensorboard, gdown, ignite, torchvision, itk, tqdm, lmdb, psutil
    # or pip install git+https://github.com/Project-MONAI/MONAI#egg=MONAI
monai.config.print_config()

import nibabel

from monai.apps import DecathlonDataset

dataset = DecathlonDataset(root_dir="./", task="Task05_Prostate", section="training", transform=None, download=True)
print(f"\nnumber of subjects: {len(dataset)}.\nThe first element in the dataset is {dataset[0]}.")

from monai.transforms import LoadNifti

loader = LoadNifti(image_only=True)
img_array = loader("Task05_Prostate/imagesTr/prostate_02.nii.gz")
print(img_array.shape)

# compose the transforms
from monai.transforms import LoadNiftiD

dict_loader = LoadNiftiD(keys=("image", "label"))
data_dict = dict_loader({"image": "Task05_Prostate/imagesTr/prostate_02.nii.gz", 
                         "label": "Task05_Prostate/labelsTr/prostate_02.nii.gz"})
print(f"image shape: {data_dict['image'].shape}, \nlabel shape: {data_dict['label'].shape}")

