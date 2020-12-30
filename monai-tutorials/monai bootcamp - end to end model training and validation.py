# title: monai bootcamp - end to end model training and validation
# author: Sun Yeop Lee

'''
This notebook takes you through the end to end workflow using for training a deep learning model. You'll do the following:

Download the MedNIST Dataset
Explore the data
Prepare training, validation, and test datasets
Use MONAI transforms, dataset, and dataloader
Define network, optimizer, and loss function
Train your model with a standard pytorch training loop
Plot your training metrics
Evaluate your model on a test set
Understand your results
Make some improvements
Revisit model training using ignite and MONAI features
Sort out problems limiting reproducability
Rework dataset partitioning
'''

import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import PIL

import torch

import monai

from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import compute_roc_auc
from monai.networks.nets import densenet121
from monai.transforms import (
    AddChannel,
    Compose,
    LoadPNG,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
)
from monai.utils import set_determinism

print_config()


# -- set up temporary data directory
# directory = os.environ.get("MONAI_DATA_DIRECTORY")
# root_dir = tempfile.mkdtemp() if directory is None else directory
# print(root_dir)

# -- download data
# resource = "https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz?dl=1"
# md5 = "0bc7306e7427e00ad1c5526a6677552d"

root_dir = r'C:\Users\sunyp\Desktop\딥노이드\Python\github repository\deepnoid-practices\monai-tutorials\data'
# compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
data_dir = os.path.join(root_dir, "MedNIST")
# if not os.path.exists(data_dir):
#     download_and_extract(resource, compressed_file, root_dir, md5)

# -- set deterministic training for reproducibility
## set_determinism will set the random seeds in both Numpy and PyTorch to ensure reproducibility. 
## seed of 0 is a bad thing to do. In general, seeds should contain a reasonable number of binary 1's and small numbers don't have them
set_determinism(seed=0) 

# -- import data 
class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
num_class = len(class_names)
image_files = [
    [
        os.path.join(data_dir, class_names[i], x)
        for x in os.listdir(os.path.join(data_dir, class_names[i]))
    ]
    for i in range(num_class)
]
num_each = [len(image_files[i]) for i in range(num_class)]
image_files_list = []
image_class = []
for i in range(num_class):
    image_files_list.extend(image_files[i])
    image_class.extend([i] * num_each[i])
num_total = len(image_class)
image_width, image_height = PIL.Image.open(image_files_list[0]).size

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names}")
print(f"Label counts: {num_each}")

# -- visualize some images
plt.subplots(3, 3, figsize=(8, 8))
for i, k in enumerate(np.random.randint(num_total, size=9)):
    im = PIL.Image.open(image_files_list[k])
    arr = np.array(im)
    plt.subplot(3, 3, i + 1)
    plt.xlabel(class_names[image_class[k]])
    plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
plt.tight_layout()
plt.show()

# -- prepare training, validation, and test data
# you'll note here that the number of images in each group changes each time this is run. Further down the notebook
# you can see a method for partitioning data that ensures the same number of images in each group each time the cell is run
val_frac = 0.1
test_frac = 0.1
train_x = list()
train_y = list()
val_x = list()
val_y = list()
test_x = list()
test_y = list()

for i in range(num_total):
    rann = np.random.random()
    if rann < val_frac:
        val_x.append(image_files_list[i])
        val_y.append(image_class[i])
    elif rann < test_frac + val_frac:
        test_x.append(image_files_list[i])
        test_y.append(image_class[i])
    else:
        train_x.append(image_files_list[i])
        train_y.append(image_class[i])

print(f"Training count: {len(train_x)}, Validation count: {len(val_x)}, Test count: {len(test_x)}")

# -- define monai transforms, dataset, dataloader to pre-process data
train_transforms = Compose(
    [
        LoadPNG(image_only=True),
        AddChannel(),
        ScaleIntensity(),
        RandRotate(range_x=15, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        ToTensor(),
    ]
)

val_transforms = Compose([LoadPNG(image_only=True), AddChannel(), ScaleIntensity(), ToTensor()])

# -- initialize the datasets and loaders for training, validation, and test sets
batch_size = 128
num_workers = 0

class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


train_ds = MedNISTDataset(train_x, train_y, train_transforms)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_ds = MedNISTDataset(val_x, val_y, val_transforms)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

test_ds = MedNISTDataset(test_x, test_y, val_transforms)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)


# -- define network and optimizer
learning_rate = 1e-5

device = torch.device("cuda:0")#if torch.cuda.is_available() )
net = densenet121(spatial_dims=2, in_channels=1, out_channels=num_class).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), learning_rate)

# -- network training
'''
We are hand-rolling a basic pytorch training loop here:

standard pytorch training loop
step through each training epoch, running through the training set in batches
after each epoch, run a validation pass, evaluating the network
if it shows improved performance, save out the model weights
later we will revisit training loops in a more Ignite / MONAI fashion
'''
epoch_num = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()

for epoch in range(epoch_num):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{epoch_num}")

    epoch_loss = 0
    step = 1

    steps_per_epoch = len(train_ds) // train_loader.batch_size

    # put the network in train mode; this tells the network and its modules to
    # enable training elements such as normalisation and dropout, where applicable
    net.train()
    for batch_data in train_loader:

        # move the data to the GPU
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)

        # prepare the gradients for this step's back propagation
        optimizer.zero_grad()
        
        # run the network forwards
        outputs = net(inputs)
        
        # run the loss function on the outputs
        loss = loss_function(outputs, labels)
        
        # compute the gradients
        loss.backward()
        
        # tell the optimizer to update the weights according to the gradients
        # and its internal optimisation strategy
        optimizer.step()

        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size + 1}, training_loss: {loss.item():.4f}")
        step += 1

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    # after each epoch, run our metrics to evaluate it, and, if they are an improvement,
    # save the model out
    
    # switch off training features of the network for this pass
    net.eval()

    # 'with torch.no_grad()' switches off gradient calculation for the scope of its context
    with torch.no_grad():
        # create lists to which we will concatenate the the validation results
        images = list()
        labels = list()

        # iterate over each batch of images and run them through the network in evaluation mode
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)

            # run the network
            val_pred = net(val_images)

            images.append(val_pred)
            labels.append(val_labels)

        # concatenate the predicted labels with each other and the actual labels with each other
        y_pred = torch.cat(images)
        y = torch.cat(labels)

        # we are using the area under the receiver operating characteristic (ROC) curve to determine
        # whether this epoch has improved the best performance of the network so far, in which case
        # we save the network in this state
        auc_metric = compute_roc_auc(y_pred, y, to_onehot_y=True, softmax=True)
        metric_values.append(auc_metric)
        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)
        if auc_metric > best_metric:
            best_metric = auc_metric
            best_metric_epoch = epoch + 1
            torch.save(net.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
            print("saved new best metric network")
        print(
            f"current epoch: {epoch + 1} current AUC: {auc_metric:.4f} /"
            f" current accuracy: {acc_metric:.4f} best AUC: {best_metric:.4f} /"
            f" at epoch: {best_metric_epoch}"
        )

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")


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
net.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
net.eval()
y_true = list()
y_pred = list()
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = (
            test_data[0].to(device),
            test_data[1].to(device),
        )
        pred = net(test_images).argmax(dim=1)
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())
            


# -- classification reprot
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# -- confusion matrix
from sklearn.metrics import confusion_matrix
cmat = confusion_matrix(y_true, y_pred)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion_matrix(y_true, y_pred), cmap="terrain", interpolation='nearest')
fig.colorbar(cax)

ax.set_xticklabels(['']+class_names, rotation=270)
ax.set_yticklabels(['']+class_names)

plt.show()


# -- issues with determinism
'''
MONAI provides monai.utils.set_determinism for replicable training
Easy to accidentally defeat, especially in a jupyter / IPython notebook
How many uses of numpy.random's underlying global instance does this notebook have?
Dataset partitioning
Image previewing
Transforms
MONAI transforms with randomised behaviour can be given / told to create their own internal numpy.random.RandomState instances
'''

# setting up transforms, revisited
## using .set_random_state allows us to pass either a seed or a numpy.random.RandomState instance
## (this is an individual instance of the numpy random number generator rather than using the global instance)
## this means that no other calls to numpy.random affect the behaviour of the Rand*** transforms
rseed = 12345678

train_transforms = Compose(
    [
        LoadPNG(image_only=True),
        AddChannel(),
        ScaleIntensity(),
        RandRotate(range_x=15, prob=0.5, keep_size=True).set_random_state(rseed),
        RandFlip(spatial_axis=0, prob=0.5).set_random_state(rseed),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5).set_random_state(rseed),
        ToTensor(),
    ]
)

val_transforms = Compose([LoadPNG(image_only=True), AddChannel(), ScaleIntensity(), ToTensor()])

# -- improving dataset partitioning
from math import floor

# make this selection deterministic and controllable by seed

dataset_seed = 12345678
r = np.random.RandomState(dataset_seed)

# calculate the number of images we want for the validation and test groups
validation_proportion = 0.1
test_proportion = 0.1
validation_count = floor(validation_proportion * num_total)
test_count = floor(test_proportion * num_total)

groups = np.zeros(num_total, dtype=np.int32)

# set the appropriate number of '1's for the validation dataset
groups[:validation_count] = 1

# then set the appropriate number of '2's for the test dataset
groups[validation_count:validation_count + test_count] = 2

# Shuffle the sequence so that 
r.shuffle(groups)

image_sets = list(), list(), list()
label_sets = list(), list(), list()

for n in range(num_total):
    image_sets[groups[n]].append(image_files_list[n])
    label_sets[groups[n]].append(image_class[n])
    
train_x, val_x, test_x = image_sets
train_y, val_y, test_y = label_sets
print(len(train_x), len(val_x), len(test_x))


# delete the temp directory
if directory is None:
    shutil.rmtree(root_dir)

