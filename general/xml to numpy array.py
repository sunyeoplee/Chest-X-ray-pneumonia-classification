import os, glob, cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

XML_DIR = 'data/xml/syndesmophyte' #Syndesmophyte
IMAGE_DIR = 'data/images'
train_images, val_images, test_images = os.listdir('data/images/train'), os.listdir('data/images/val'), os.listdir('data/images/test')
def which(image_file):
    if image_file in train_images:
        return 'train'
    elif image_file in val_images:
        return 'val'
    elif image_file in test_images:
        return 'test'
    return

for xml in os.listdir(XML_DIR):
    path = os.path.join(XML_DIR, xml)
    tree = ET.parse(path)
    root = tree.getroot()
    polygons = []
    classes = []
    for ele in root:
        image = ele.attrib['image']
        print(image)
        image_file = image.split('.')[0]+'.png'
        category = which(image_file)
        if not category:
            continue
        imread = plt.imread(os.path.join(IMAGE_DIR, category, image_file))
        for subele in ele:
            if subele.tag=='Annotation':
                classes.append(subele.get('class'))
                polygon = []
                for subsub in subele:
                    for subsubsub in subsub:
                        polygon.append((float(subsubsub.get('x')), float(subsubsub.get('y'))))
                polygons.append(polygon)
    h, w = imread.shape[:2]
    os.makedirs(f'data/masks/syndesmophyte/{​​​​​​​xml.split(".")[0]}​​​​​​​', exist_ok=True)
    for polygon, classe in zip(polygons, classes):
        img = Image.new('L', (w, h), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        array = np.array(img)
        new_class = classe.split()[0]+'U' if classe[-1]=='A' else classe.split()[0]+'L'
        print('\t', classe, new_class, array.shape, array.dtype, array.max(), array.min())
        cv2.imwrite(f'data/masks/syndesmophyte/{​​​​​​​xml.split(".")[0]}​​​​​​​/{​​​​​​​new_class}​​​​​​​.png', array)
