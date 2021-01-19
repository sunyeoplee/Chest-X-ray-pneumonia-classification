# title: monai bootcamp - datasets
# author: Sun Yeop Lee

'''
Overview
This notebook introduces you to the MONAI dataset APIs:

Recap the base dataset API
Understanding the caching mechanism
Dataset utilities
'''
import os
import time
import torch

import monai
monai.config.print_config()

# -- Monai dataset
'''
A MONAI Dataset is a generic dataset with a __len__ property, __getitem__ property, and an optional callable data transform when fetching a data sample.

We'll start by initializing some generic data, calling the Dataset class with the generic data, and specifying None for our transforms.
'''
## dataset
items = [{"data": 4}, 
         {"data": 9}, 
         {"data": 3}, 
         {"data": 7}, 
         {"data": 1},
         {"data": 2},
         {"data": 5}]
dataset = monai.data.Dataset(items, transform=None)

print(f"Length of dataset is {len(dataset)}")
for item in dataset:
    print(item)

## dataloader
for item in torch.utils.data.DataLoader(dataset, batch_size=2):
    print(item)

## load items with a customized transform
class SquareIt(monai.transforms.MapTransform):
    """a simple transform to return a squared number"""

    def __init__(self, keys):
        monai.transforms.MapTransform.__init__(self, keys)
        print(f"keys to square it: {self.keys}")

        
    def __call__(self, x):
        key = self.keys[0]
        data = x[key]
        output = {key: data ** 2}
        return output

square_dataset = monai.data.Dataset(items, transform=SquareIt(keys='data'))
for item in square_dataset:
    print(item)

'''
Keep in mind

SquareIt is implemented as creating a new dictionary output instead of overwriting the content of dict x directly. So that we can repeatedly apply the transforms, for example, in multiple epochs of training
SquareIt.__call__ read the key information from self.keys but does not write any properties to self. Because writing properties will not work with a multi-processing data loader.
In most of the MONAI preprocessing transforms, we assume x[key] has the shape: (num_channels, spatial_dim_1, spatial_dim_2, ...). The channel dimension is not omitted even if num_channels equals to 1, but the spatial dimensions could be omitted.
'''

# -- monai data caching
class SlowSquare(monai.transforms.MapTransform):
    """a simple transform to slowly return a squared number"""
  
    def __init__(self, keys):
        monai.transforms.MapTransform.__init__(self, keys)
        print(f"keys to square it: {self.keys}")

    def __call__(self, x):
        time.sleep(1.0)
        output = {key: x[key] ** 2 for key in self.keys}
        return output

square_dataset = monai.data.Dataset(items, transform=SlowSquare(keys='data'))

start_time = time.time()
list(item for item in square_dataset)
end_time = time.time()
print('total time =',end_time - start_time)


## cache dataset
'''
When using CacheDataset the caching is done when the object is initialized for the first time, so the initialization is slower than a regular dataset.

By caching the results of non-random preprocessing transforms, it accelerates the training data pipeline. If the requested data is not in the cache, all transforms will run normally.
To improve the caching efficiency, always put as many as possible non-random transforms before the randomized ones when composing the chain of transforms.
'''

square_cached = monai.data.CacheDataset(items, transform=SlowSquare(keys='data'))

start_time = time.time()
list(item for item in square_cached)
end_time = time.time()
print('total time =',end_time - start_time)

## persistent caching
'''
PersistantDataset allows for persistent storage of pre-computed values to efficiently manage larger than memory dictionary format data.

The non-random transform components are computed when first used and stored in the cache_dir for rapid retrieval on subsequent uses.
'''
square_persist = monai.data.PersistentDataset(items, transform=SlowSquare(keys='data'), cache_dir="my_cache")

## The caching happens at the first epoch of loading the dataset, so calling the dataset the first time should take about 7 seconds.start_time = time.time()
start_time = time.time()
list(item for item in square_persist)
end_time = time.time()
print('total time =', end_time - start_time)

## During the initialization of the PersistentDataset we passed in the parameter "my_cache" for the location to store the intermediate data. We'll look at that directory below.
os.listdir('my_cache')

## When calling out to the dataset on the following epochs, it will not call the slow transform but used the cached data.
start_time = time.time()
list(item for item in square_persist)
end_time = time.time()
print('total time =', end_time - start_time)

## Fresh dataset instances can make use of the caching data
square_persist_1 = monai.data.PersistentDataset(items, transform=SlowSquare(keys='data'), cache_dir="my_cache")
start_time = time.time()
list(item for item in square_persist)
end_time = time.time()
print('total time =', end_time - start_time)
'''
Caching in action
There's also a SmartCacheDataset to hide the transforms latency with less memory consumption.
The dataset tutorial notebook has a working example and a comparison of different caching mechanism in MONAI: https://github.com/Project-MONAI/tutorials/blob/master/acceleration/dataset_type_performance.ipynb
'''

# -- Other dataset utilities

## ZipDataset
'''
ZipDataset will zip several PyTorch datasets and output data(with the same index) together in a tuple. If a single dataset's output is already a tuple, flatten it and extend to the result. It supports applying some transforms on the associated new element.
'''

items = [4, 9, 3]
dataset_1 = monai.data.Dataset(items)

items = [7, 1, 2, 5]
dataset_2 = monai.data.Dataset(items)

def concat(data):
    # data[0] is an element from dataset_1
    # data[1] is an element from dataset_2
    return (f"{data[0]} + {data[1]} = {data[0] + data[1]}",)

zipped_data = monai.data.ZipDataset([dataset_1, dataset_2], transform=concat)
for item in zipped_data:     
    print(item)

## Common Datasets
'''
MONAI provides access to some commonly used medical imaging datasets through DecathlonDataset. This function leverages the features described throughout this notebook.
'''
dataset = monai.apps.DecathlonDataset(root_dir="./", task="Task04_Hippocampus", section="training", download=True)
print(dataset.get_properties("numTraining"))
print(dataset.get_properties("description"))

print(dataset[0]['image'].shape)
print(dataset[0]['label'].shape)









