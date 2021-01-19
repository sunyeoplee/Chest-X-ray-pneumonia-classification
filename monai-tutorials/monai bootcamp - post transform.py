'''
title: monai bootcamp - post transform
author: Sun Yeop Lee
'''
'''
Overview
This notebook introduces you to the MONAI APIs for:

sliding window inference
post-processing transforms
'''

import torch
import matplotlib.pyplot as plt

import monai
monai.config.print_config()
from monai.inferers import sliding_window_inference


# -- sliding window inference

## a toy model for inference
class ToyModel:
    # A simple model generates the output by adding an integer `pred` to input.
    # each call of this instance increases the integer by 1.
    pred = 0
    def __call__(self, input):
        self.pred = self.pred + 1
        return input + self.pred

## run the inference using sliding window

input_tensor = torch.zeros(1, 1, 200, 200)
output_tensor = sliding_window_inference(
    inputs=input_tensor, 
    predictor=ToyModel(), 
    roi_size=(40, 40), 
    sw_batch_size=1, 
    overlap=0.5, 
    mode="constant")
plt.imshow(output_tensor[0, 0])
plt.show()


## Gaussian weighted windows
'''
For a given input image window, the convolutional neural networks often predict the central regions more accurately than the border regions, usually due to the stacked convolutions' receptive field.

Therefore, it is worth considering a "Gaussian weighted" prediction to emphasize the central region predictions when we stitch the windows into a complete inference output.

The following is an example of a 40x40-pixel Gaussian window map constructed using GaussianFilter from MONAI. 
This is also integrated into the sliding window module.
'''
win_size = (40, 40)
gaussian = torch.zeros(win_size, device="cpu")
center_coords = [i // 2 for i in win_size]
sigmas = [i * 0.125 for i in win_size]
gaussian[tuple(center_coords)] = 1
pt_gaussian = monai.networks.layers.GaussianFilter(len(win_size), sigma=sigmas).to(device="cpu", dtype=torch.float)
gaussian = pt_gaussian(gaussian.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
print(gaussian.shape)
plt.imshow(gaussian)
plt.show()



input_tensor = torch.zeros(1, 1, 200, 200)
output_tensor_1 = sliding_window_inference(
    inputs=input_tensor, 
    predictor=ToyModel(), 
    roi_size=(40, 40), 
    sw_batch_size=1, 
    overlap=0.5, 
    mode="gaussian")
plt.imshow(output_tensor_1[0, 0])
plt.show()

plt.subplots(1, 2)
plt.subplot(1, 2, 1); plt.imshow(output_tensor[0, 0])
plt.subplot(1, 2, 2); plt.imshow(output_tensor_1[0, 0])
plt.show()

## -- post-processing transforms
'''
This section will set up and load a SegResNet model, run sliding window inference, and post-process the model output volumes:

Argmax to get a discrete prediction map
Remove small isolated predicted regions
Convert the segmentation regions into contours
We'll start by importing all of our dependencies.
'''
import os
import glob

from monai.apps import download_and_extract
from monai.utils import set_determinism
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import SegResNet
from monai.transforms import (
    AddChanneld,
    AsDiscrete,
    Compose,
    CropForegroundd,
    KeepLargestConnectedComponent,
    LabelToContour,
    LoadNiftid,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

root_dir = r'C:\Users\sunyp\Desktop\딥노이드\Python\github repository\deepnoid-practices\monai-tutorials\data'
compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
data_dir = os.path.join(root_dir, "Task09_Spleen")
download_and_extract(resource, compressed_file, root_dir, md5)

## set up the validation data, preprocessing transforms, and data loader
images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(images, labels)
]
val_files = data_dicts[-9:]

val_transforms = Compose(
    [
        LoadNiftid(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

## set up the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
model = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
).to(device)

model_path = os.path.join(root_dir, "segresnet_model_epoch30.pth")
model.load_state_dict(torch.load(model_path))
model.eval()
print(f"model from {model_path}.")

## run the sliding window inferenc
val_data = next(iter(val_loader))
val_data = val_data["image"].to(device)

roi_size = (160, 160, 160)
sw_batch_size = 4
with torch.no_grad():
  val_output = sliding_window_inference(val_data, roi_size, sw_batch_size, model)
print(val_output.shape, val_output.device)
slice_idx = 80
plt.title(f"image -- slice {slice_idx}")
plt.imshow(val_output.detach().cpu()[0, 1, :, :, 80], cmap="gray")


roi_size = (88, 88, 88)
sw_batch_size = 1
with torch.no_grad():
  val_output = sliding_window_inference(
      val_data, roi_size, sw_batch_size=sw_batch_size, predictor=model, mode="gaussian", overlap=0.2)
print(val_output.shape, val_output.device)

slice_idx = 80
plt.title(f"image -- slice {slice_idx}")
plt.imshow(val_output.detach().cpu()[0, 1, :, :, 80], cmap="gray")

## post processing: argmax over the output probabilities into a discrete map
argmax = AsDiscrete(argmax=True)(val_output)
print(argmax.shape)

slice_idx = 80
plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plt.title(f"image -- slice {slice_idx}")
plt.imshow(val_data.detach().cpu()[0, 0, :, :, 80], cmap="gray")

plt.subplot(1, 2, 2)
plt.title(f"argmax -- slice {slice_idx}")
plt.imshow(argmax.detach().cpu()[0, 0, :, :, 80])

## post processing: connected component analysis to select the largest segmentation region

largest = KeepLargestConnectedComponent(applied_labels=[1])(argmax)
print(largest.shape)

slice_idx = 80
plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plt.title(f"image -- slice {slice_idx}")
plt.imshow(val_data.detach().cpu()[0, 0, :, :, 80], cmap="gray")

plt.subplot(1, 2, 2)
plt.title(f"largest component -- slice {slice_idx}")
plt.imshow(largest.detach().cpu()[0, 0, :, :, 80])

## post-processing: convert the region into a contour map
contour = LabelToContour()(largest)
print(contour.shape)

slice_idx = 80
plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plt.title(f"image -- slice {slice_idx}")
plt.imshow(val_data.detach().cpu()[0, 0, :, :, 80], cmap="gray")

plt.subplot(1, 2, 2)
plt.title(f"contour -- slice {slice_idx}")
plt.imshow(contour.detach().cpu()[0, 0, :, :, 80], cmap="Greens")

map_image = contour + val_data

slice_idx = 80
plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plt.title(f"image -- slice {slice_idx}")
plt.imshow(val_data.detach().cpu()[0, 0, :, :, 80], cmap="gray")

plt.subplot(1, 2, 2)
plt.title(f"contour -- slice {slice_idx}")
plt.imshow(map_image.detach().cpu()[0, 0, :, :, 80], cmap="gray")

from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter

with SummaryWriter(log_dir=root_dir) as writer:
    plot_2d_or_3d_image(map_image, step=0, writer=writer, tag="segmentation")
    plot_2d_or_3d_image(val_output, step=0, max_channels=2, writer=writer, tag="Probability")




















