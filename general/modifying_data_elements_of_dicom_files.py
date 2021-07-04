# title: modifying data elements of dicom files
# author: Sun Yeop Lee

# import libraries
import sys
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import pydicom
import os
import glob
import pandas as pd

import scipy.ndimage 
from skimage import measure, morphology # conda install scikit-image


import difflib

# define functions
def count_dicom_files(path):
    n = 0 
    for file in os.listdir(path):
        if file.endswith('.dcm'):
            n = n + 1
    return n

def load_scan(path):
    imagelist = []
    for (dir, subdirlist, filelist) in os.walk(path, topdown=True):
        for filename in filelist:
            if ".dcm" in filename.lower():
                imagelist.append(os.path.join(dir, filename))
    return imagelist

def plot_pixel_array(dataset):
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show(block=False)
    plt.pause(3)
    plt.close()



# import dicom images
influenza_folder = r'C:\Users\sunyp\Desktop\딥노이드\연구 프로젝트\부산대학교 COVID19\데이터\Influenza(부산대병원)'
influenza_filelist = os.listdir(influenza_folder)

# count the number of dicom files
print(count_dicom_files(influenza_folder))
print(len(influenza_filelist))

# compare the correct and wrong images
testimage_correct = pydicom.dcmread(os.path.join(influenza_folder, os.listdir(influenza_folder)[1]))
plot_pixel_array(testimage_correct)
print(testimage_correct.PhotometricInterpretation)
# alternative ways to check data elements
# testimage_correct[0x0028,0x0004]
# testimage_correct['PhotometricInterpretation']

testimage_wrong = pydicom.dcmread(os.path.join(influenza_folder, os.listdir(influenza_folder)[2]))
plot_pixel_array(testimage_wrong)
print(testimage_wrong.PhotometricInterpretation)

# invert pixel of the wrong image
pixel_to_modify = testimage_wrong.pixel_array
pixel_min = np.min(pixel_to_modify)
pixel_max = np.max(pixel_to_modify)
pixel_modified = np.invert(pixel_to_modify)
print(np.max(pixel_modified), np.min(pixel_modified))
testimage_wrong.PixelData = pixel_modified.tobytes()

    
# check the inversion was done correctly
plot_pixel_array(testimage_wrong)

# apply the inversion to the dicom files with MONOCHROME1 and switch them to MONOCHROME2
for file in influenza_filelist:
    image = pydicom.dcmread(os.path.join(influenza_folder, file))
    if image.PhotometricInterpretation == 'MONOCHROME1':
        image.PhotometricInterpretation = 'MONOCHROME2' # convert photometric interpretation
        image.PixelData = np.invert(image.pixel_array).tobytes() # invert pixel values
        new_folder = r'C:\Users\sunyp\Desktop\딥노이드\연구 프로젝트\부산대학교 COVID19\데이터\Influenza(부산대병원)_inverted'
        image.save_as(os.path.join(new_folder, file))
    else:
        image.save_as(os.path.join(new_folder, file))

# check the inversion was done correctly
test = pydicom.dcmread(os.path.join(new_folder, os.listdir(new_folder)[2]))
plot_pixel_array(test)






