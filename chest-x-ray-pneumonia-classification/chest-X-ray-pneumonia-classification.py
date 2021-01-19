# title: data preparation
# author: Sun Yeop Lee

# -- set up environment
import os
from glob import glob

import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
import torchvision 
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.__version__
device_cnt = torch.cuda.device_count()
cur_device = torch.cuda.current_device()
device_name = torch.cuda.get_device_name(cur_device)
print(f"Number of CUDA-enabled GPUs: {device_cnt}\nCurrent Device ID: {cur_device}\nCurrent Device Name: {device_name}")
device.type

from PIL import Image 
import monai
print(monai.__version__)
from monai.config import print_config
print_config()
from monai.transforms import Compose, LoadPNG, AddChannel, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom, \
    NormalizeIntensity, SpatialPad, SqueezeDim, Resize
from monai.networks.nets import densenet121
from monai.metrics import compute_roc_auc

import matplotlib.pyplot as plt

root_dir = r'C:\Users\sunyp\Desktop\딥노이드\Python\github repository\deepnoid-practices\chest-x-ray-pneumonia-classification'
os.chdir(root_dir)

# -- set up hyperparameters
seed = 1
torch.manual_seed(seed)
momentum = 0.5
batch_size = 128
val_interval = 5
n_epoch = 100


# -- load data
data_dir = 'C:\\Users\\sunyp\\Desktop\\딥노이드\\Python\\data\\03. Chest X-Ray_PNEUMONIA'
class_names = sorted([x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))])
n_class = len(class_names)
image_files_by_class = [[os.path.join(data_dir, class_name, x) # a list of paths to images separated by class
                            for x in os.listdir(os.path.join(data_dir, class_name))
                            if x.endswith('.jpeg')] 
                            for class_name in class_names]

image_file_list = [] # paths to all images
image_label_list = [] # paths to all labels
for i, class_name in enumerate(class_names):
    image_file_list.extend(image_files_by_class[i])
    image_label_list.extend([i] * len(image_files_by_class[i]))


# -- inspect data
## count data
classes_names = os.listdir(data_dir)
n_class = len(classes_names)
print('The number of classes:', n_class)

n_total = len(image_file_list)
n_normal = len( os.listdir(os.path.join(data_dir, classes_names[0])))
n_pneumonia = len( os.listdir(os.path.join(data_dir, classes_names[1])))
print('Sample size:', 'total ({}), normal ({}), non-case ({})'.format(n_total, n_normal, n_pneumonia))

## image shape and dimension
image_width_list = [] # a list of image width
image_height_list = [] # a list of image height
image_channel_list = [] # the number of channels
image_width_list = [Image.open(x).size[0] for x in image_file_list if x.endswith('.jpeg' or '.jpg')]
image_height_list = [Image.open(x).size[1] for x in image_file_list if x.endswith('.jpeg' or '.jpg')]
# image_channel_list = [Image.open(x).size[-1] for x in image_file_list if ]


plt.hist(image_width_list)
plt.show()
max(image_width_list)

plt.hist(image_height_list)
plt.show()
max(image_height_list)

image_width, image_height = Image.open(image_file_list[0]).size
print("Image shape:", image_width, "x", image_height)



#  visualize images
plt.subplots(3, 3, figsize=(8, 8))
for i,k in enumerate(np.random.randint(n_total, size=9)):
    im = Image.open(image_file_list[k])
    arr = np.array(im)
    plt.subplot(3, 3, i + 1)
    plt.xlabel(class_names[image_label_list[k]])
    plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
plt.tight_layout()
plt.show()

# -- prepare training, validation, and test data lists
valid_frac, test_frac = 0.2, 0.1
trainX, trainY = [], []
valX, valY = [], []
testX, testY = [], []

for i in range(n_total):
    rann = np.random.random()
    if rann < valid_frac:
        valX.append(image_file_list[i])
        valY.append(image_label_list[i])
    elif rann < test_frac + valid_frac:
        testX.append(image_file_list[i])
        testY.append(image_label_list[i])
    else:
        trainX.append(image_file_list[i])
        trainY.append(image_label_list[i])

print("Training count =",len(trainX),"Validation count =", len(valX), "Test count =",len(testX))

# -- specify transformations
## define custom transformation
class np_to_grayscale:
    '''
    convert rgb to grayscale if an image is in rgb.
    '''
    def __init__(self):
        pass

    def __call__(self, image):
        if image.ndim > 2:
            tensor = torchvision.transforms.ToTensor()
            gray = torchvision.transforms.Grayscale()

            image = tensor(image)
            image = gray(image)
            image = image.numpy()
            image = np.squeeze(image)

            return image

        else: 
            return image 

# class Visualize(object):
#     def __init__(self):
#         pass

#     def __call__(self, image):
#         image.show()
#         return image

## specify transformations for each dataset
train_transforms = Compose([
    LoadPNG(image_only=True),
    np_to_grayscale(),
    AddChannel(), # needs to come early because most monai transforms expect the channel to be the first dimension
    Resize((64, 64)),
    NormalizeIntensity(),
    RandRotate(range_x=15, prob=0.5, keep_size=True),
    RandFlip(spatial_axis=0, prob=0.5),

    ToTensor()
])

val_transforms = Compose([
    LoadPNG(image_only=True),
    np_to_grayscale(),
    AddChannel(),
    Resize((64, 64)),
    NormalizeIntensity(),
    ToTensor()
])

## visualize transformations
my_transforms = Compose([
    LoadPNG(image_only=True),
    np_to_grayscale(),
    AddChannel(), # needs to come early because most monai transforms expect the channel to be the first dimension
    Resize((64, 64)),
    # NormalizeIntensity(),
    # RandRotate(range_x=15, prob=0.5, keep_size=True),
    # RandFlip(spatial_axis=0, prob=0.5),
    ToTensor(),
    SqueezeDim()
])

example = my_transforms(trainX[1])
plt.imshow(example, 'gray')
plt.show()



class chestxray_dataset(Dataset):
    
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

train_ds = chestxray_dataset(trainX, trainY, train_transforms)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)

val_ds = chestxray_dataset(valX, valY, val_transforms)
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

test_ds = chestxray_dataset(testX, testY, val_transforms)
test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)



# -- define network and optimizer
model = densenet121(
    spatial_dims=2,
    in_channels=1,
    out_channels=n_class
).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)




# -- model training
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()


for epoch in range(1, 1 + n_epoch):
    print('-' * 10)
    print(f"epoch {epoch}/{n_epoch}")
    model.train()
    epoch_loss = 0
    step = 0
    for i, batch_data in enumerate(train_loader):
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
        epoch_len = len(train_ds) // train_loader.batch_size
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch} average loss: {epoch_loss:.4f}") # average over all the batches

    if epoch % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            auc_metric = compute_roc_auc(y_pred, y, to_onehot_y=True, softmax=True)
            metric_values.append(auc_metric)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            if auc_metric > best_metric:
                best_metric = auc_metric
                best_metric_epoch = epoch
                torch.save(model.state_dict(), 'best_metric_model.pth')
                print('saved new best metric model')
            print(f"current epoch: {epoch} current AUC: {auc_metric:.4f}"
                  f" current accuracy: {acc_metric:.4f} best AUC: {best_metric:.4f}"
                  f" at epoch: {best_metric_epoch}")
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

# check if the model was on cuda
next(model.parameters()).is_cuda

# -- plot the loss and metric
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val AUC")
x = [(i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.show()

# -- evaluate the model on the test dataset
