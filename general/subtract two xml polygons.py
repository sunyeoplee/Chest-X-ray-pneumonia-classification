import os, glob, cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
 
 

WHOLE_XML = 'whole.xml'
SUBTRACTING_XML = 'syndesmophyte.xml'

SUBTRACTED_XML = 'vertebral_body.xml'
 

def get_polygon(binary_image):
    contour = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0][0][:,0,:]
    contour = list(contour)+[contour[0]]
    def gradient(coor1, coor2):
        if coor1[0]==coor2[0] and coor1[1]!=coor2[1]:
            return np.inf
        return (coor2[1]-coor1[1])/(coor2[0]-coor1[0])

 

    polygon = []
    for ele in contour:
        polygon.append(tuple(ele))
        if len(polygon)>2:
            if any([polygon[-3][i]==polygon[-2][i]==polygon[-1][i] for i in range(len(ele))]):
                polygon.pop(-2)
            if gradient(polygon[-3], polygon[-2])==gradient(polygon[-2], polygon[-1]):
                polygon.pop(-2)
    return polygon

 

def create_mask_subtracted_xml(whole_xml, subtracting_xml):
    xmls = [whole_xml, subtracting_xml]
    trees = [ET.parse(xml) for xml in xmls]
    roots = [tree.getroot() for tree in trees]
    
    polygons = []
    image = None
    colors = []
    for root in roots:
        for ele in root:
            image = ele.attrib['image']
            imread = plt.imread(glob.glob(image.split('.')[0]+'.*')[0])
            polygon_list = []
            for subele in ele:
                if subele.get('color'):
                    colors.append(subele.get('color'))
                if subele.get('type')=='polygon':
                    polygon = []
                    for subsub in subele:
                        for subsubsub in subsub:
                            polygon.append((float(subsubsub.get('x')), float(subsubsub.get('y'))))
                    polygon_list.append(polygon)
            polygons.append(polygon_list)
    h, w = imread.shape[:2]
                
    def get_mask(polygon_list):
        masks = []
        for polygon in polygon_list:
            img = Image.new('L', (w, h), 0)
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            masks.append(np.array(img))
        return np.logical_or(*masks).astype(np.uint8) if len(masks)>1 else masks[0]
    masks = [get_mask(polygon_list) for polygon_list in polygons]
    subtraction = masks[0]-masks[0]*masks[1]
    polygon = get_polygon(subtraction)
    
    data = ET.Element(roots[0].tag)
    data.text = roots[0].text
    data.tail = roots[0].tail
    for ele in roots[0]:
        sub = ET.SubElement(data, ele.tag)
        [sub.set(key, ele.attrib[key]) for key in ele.attrib.keys()]
        sub.text = ele.text
        sub.tail = ele.tail
        for subele in ele:
            sub2 = ET.SubElement(sub, subele.tag)
            [sub2.set(key, subele.attrib[key]) for key in subele.attrib.keys()]
            if sub2.get('color'):
                sub2.set('color', str(abs(int(colors[0])-int(colors[1]))))
            sub2.text = subele.text
            sub2.tail = subele.tail
            for subsub in subele:
                sub3 = ET.SubElement(sub2, subsub.tag)
                sub3.text = subsub.text
                sub3.tail = subsub.tail
                for coor in polygon:
                    sub4 = ET.SubElement(sub3, 'Coordinate')
                    sub4.tail = subsub.tail+' '*4
                    sub4.set('x', str(coor[0]))
                    sub4.set('y', str(coor[1]))
                sub4.tail = subsub.tail+' '*2
    data_string = ET.tostring(data).decode()
    print(data_string)
    xml = open(SUBTRACTED_XML, "w")
    xml.write(data_string)
    return data_string
 
 
create_mask_subtracted_xml(WHOLE_XML, SUBTRACTING_XML)