# title: brain tumor segmentation
# author: Sun Yeop Lee

# -- import libraries and set up environment
import tempfile
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import monai
from monai.utils import first
from monai.data import Dataset, ArrayDataset, create_test_image_3d, DataLoader
from monai.transforms import (
    Transform,
    Randomizable,
    AddChannel,
    Compose,
    LoadNifti,
    Lambda,
    RandSpatialCrop,
    RandSpatialCropd,
    ToTensor,
)


# -- create a temp directory and add some Nifti files containing random spheres.
root_dir = tempfile.mkdtemp()
print(root_dir)
filenames = []

for i in range(5):
    im, _ = create_test_image_3d(256, 256, 256)

    filename = f"{root_dir}/im{i}.nii.gz"
    filenames.append(filename)

    n = nib.Nifti1Image(im, np.eye(4))
    nib.save(n, filename)


# -- array transforms
## load nifti file
trans = LoadNifti()

img, header = trans(filenames[0])

print(img.shape, header["filename_or_obj"])
plt.imshow(img[128])
plt.show()

## use Compose to create a sequence of operations 
trans = Compose([LoadNifti(image_only=True), AddChannel()])

img = trans(filenames[0])

print(img.shape)

## convert to tensor
trans = Compose([LoadNifti(image_only=True), AddChannel(), ToTensor()])

img = trans(filenames[0])

print(type(img), img.shape, img.get_device())

## define a custom transform operation using lambda
def sum_width(img):
    return img.sum(1)

trans = Compose([LoadNifti(image_only=True), AddChannel(), Lambda(sum_width)])

img = trans(filenames[0])

plt.imshow(img[0])
plt.show()

## define a custom transform operation by creating a subclass of Transform. This way, you can define attriutes. 
class SumDimension(Transform):
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, inputs):
        return inputs.sum(self.dim)


trans = Compose([LoadNifti(image_only=True), AddChannel(), SumDimension(2)])

img = trans(filenames[0])

plt.imshow(img[0])

## to define a transforms with stochastic operation, you can inherit from Randomizable.
## it is used to randomize variables and distinguish from deterministic transforms. 
## Randomizable.set_random_state() can be used to control the randomization process.
class RandAdditiveNoise(Randomizable, Transform):
    def __init__(self, prob: float = 0.5, max_add: float = 1.0) -> None:
        self.prob = np.clip(prob, 0.0, 1.0)
        self.max_add = max_add
        self._noise = 0

    def randomize(self, data: np.ndarray) -> None:
        self._noise = 0

        if self.R.random() < self.prob:
            noise_array = self.R.rand(*data.shape[1:])[None]
            self._noise = (noise_array * self.max_add).astype(data.dtype)

    def add_noise(self, img: np.ndarray) -> np.ndarray:
        return img + self._noise

    def __call__(self, img: np.ndarray) -> np.ndarray:
        self.randomize(img)
        return self.add_noise(img)


trans = Compose([LoadNifti(image_only=True), AddChannel(), RandAdditiveNoise()])

img = trans(filenames[0])

plt.imshow(img[0, 128])
plt.show()

## -- dictionary transforms 
## for a pipeline with multiple values, transforms can operate on dictionaries of arrays.
fn_keys = ("img", "seg")  # filename keys for image and seg files
filenames = []

for i in range(5):
    im, seg = create_test_image_3d(256, 256, 256)

    im_filename = f"{root_dir}/im{i}.nii.gz"
    seg_filename = f"{root_dir}/seg{i}.nii.gz"

    filenames.append({"img": im_filename, "seg": seg_filename})

    n = nib.Nifti1Image(im, np.eye(4))
    nib.save(n, im_filename)

    n = nib.Nifti1Image(seg, np.eye(4))
    nib.save(n, seg_filename)


## import dictionary equivalent transforms 
from monai.transforms import (
    MapTransform,
    AddChanneld,
    LoadNiftid,
    Lambdad,
    ToTensord,
)
##  keys in LoadNiftid states which keys contain paths to Nifti files
trans = LoadNiftid(keys=fn_keys)

data = trans(filenames[0])

print(list(data.keys()))

## Lambdad applies the given callable to each array named by keys separately.
## we can use this to define transforms operating on different named values in the dictionary at different points in the sequence.
def sum_width(img):
    return img.sum(1)


def max_width(img):
    return img.max(1)


trans = Compose(
    [
        LoadNiftid(fn_keys),
        AddChanneld(fn_keys),
        Lambdad(("img",), sum_width),  # sum the image in the width dimension
        Lambdad(("seg",), max_width),  # take max label in the width dimension
    ]
)

imgd = trans(filenames[0])
img = imgd["img"]
seg = imgd["seg"]

plt.imshow(np.hstack((img[0] * 5 / img.max(), seg[0])))
plt.show()

## adapting the array based transforms to operate over dictionaries
from monai.config import KeysCollection
from typing import Optional, Any, Mapping, Hashable


class RandAdditiveNoised(Randomizable, MapTransform):
    def __init__(
        self, keys: KeysCollection, prob: float = 0.5, max_add: float = 1.0
    ) -> None:
        super(Randomizable, self).__init__(keys)
        self.transform = RandAdditiveNoise(prob, max_add)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandAdditiveNoised":
        self.transform.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        self.transform.randomize(data)

    def __call__(
        self, data: Mapping[Hashable, np.ndarray]
    ) -> Mapping[Hashable, np.ndarray]:
        self.randomize(data[monai.utils.first(self.keys)])

        d = dict(data)
        for key in self.keys:
            d[key] = self.transform.add_noise(d[key])
        return d


trans = Compose(
    [LoadNiftid(fn_keys), AddChanneld(fn_keys), RandAdditiveNoised(("img",))]
)

img = trans(filenames[0])
print(list(img.keys()))

# note that we're adding random noise to the image alone, not the segmentation
plt.imshow(np.hstack([img["img"][0, 50], img["seg"][0, 50]]))
plt.show()

# -- loading datasets
## create a dataset object. 
images = [fn["img"] for fn in filenames]

transform = Compose([LoadNifti(image_only=True), AddChannel(), ToTensor()])

ds = Dataset(images, transform)

img_tensor = ds[0]

print(img_tensor.shape, img_tensor.get_device())

## ArrayDataset is specifically for supervised training. 
## It can accept data arrays for image separately from those for segmentations or labels with their own transforms.
images = [fn["img"] for fn in filenames]
segs = [fn["seg"] for fn in filenames]

img_transform = Compose(
    [
        LoadNifti(image_only=True),
        AddChannel(),
        RandSpatialCrop((128, 128, 128), random_size=False),
        RandAdditiveNoise(),
        ToTensor(),
    ]
)

seg_transform = Compose(
    [
        LoadNifti(image_only=True),
        AddChannel(),
        RandSpatialCrop((128, 128, 128), random_size=False),
        ToTensor(),
    ]
)

ds = ArrayDataset(images, img_transform, segs, seg_transform)

im, seg = ds[0]

plt.imshow(np.hstack([im.numpy()[0, 48], seg.numpy()[0, 48]]))
plt.show()

## alternatively, Dataset can be used with dictionary based transforms to construct a result mapping.

trans = Compose(
    [
        LoadNiftid(fn_keys),
        AddChanneld(fn_keys),
        RandAdditiveNoised(("img",)),
        RandSpatialCropd(fn_keys, (128, 128, 128), random_size=False),
        ToTensord(fn_keys),
    ]
)

ds = Dataset(filenames, trans)

item = ds[0]
im, seg = item["img"], item["seg"]

plt.imshow(np.hstack([im.numpy()[0, 48], seg.numpy()[0, 48]]))

## DataLoader is used to generate batches.
batch_size = 10

loader = DataLoader(ds, batch_size, num_workers=5)

batch = first(loader)

print(list(batch.keys()), batch["img"].shape)

f, ax = plt.subplots(2, 1, figsize=(8, 4))
ax[0].imshow(np.hstack(batch["img"][:, 0, 64]))
ax[1].imshow(np.hstack(batch["seg"][:, 0, 64]))











































