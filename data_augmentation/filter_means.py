# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:53:32 2019

@author: xl
"""

import cv2 as cv
from xml.dom.minidom import parse
import os


def read_xml(xml_file):
    dom_tree = parse(xml_file)
    root_node = dom_tree.documentElement
    box = root_node.getElementsByTagName('object')[0].getElementsByTagName('bndbox')[0]
    xmin = box.getElementsByTagName('xmin')[0].childNodes[0].data
    ymin = box.getElementsByTagName('ymin')[0].childNodes[0].data
    xmax = box.getElementsByTagName('xmax')[0].childNodes[0].data
    ymax = box.getElementsByTagName('ymax')[0].childNodes[0].data
    return int(xmin), int(ymin), int(xmax), int(ymax)


def update_xml(xml_file, xmin, ymin, xmax, ymax):
    dom_tree = parse(xml_file)
    root_node = dom_tree.documentElement
    box = root_node.getElementsByTagName('object')[0].getElementsByTagName('bndbox')[0]
    box.getElementsByTagName('xmin')[0].childNodes[0].data = str(xmin)
    box.getElementsByTagName('ymin')[0].childNodes[0].data = str(ymin)
    box.getElementsByTagName('xmax')[0].childNodes[0].data = str(xmax)
    box.getElementsByTagName('ymax')[0].childNodes[0].data = str(ymax)
    with open(xml_file, 'w', encoding='utf-8') as f:
        dom_tree.writexml(f, addindent='  ', encoding='utf-8')


if __name__ == '__main__':
    num_train = 800
    for index in range(num_train):
        img_path = r'F:\anaconda\project\pattren_1\train\image\train_' + (str(index + 1)).zfill(4) + '.jpg'
        xml_path = r'F:\anaconda\project\pattren_1\train\xml\train_' + (str(index + 1)).zfill(4) + '.xml'
        x1, y1, x2, y2 = read_xml(xml_path)
        img = cv.imread(img_path)   
        height, width = img.shape[:2]
        reSize1 = cv.blur(img,(10,10)) 
        img_shade_path = 'F:\\anaconda\\project\\pattren_1\\train\\image_shade\\train_' + (str(index + 4801)).zfill(4) + '.jpg'
        xml_shade_path = 'F:\\anaconda\\project\\pattren_1\\train\\xml_shade\\train_' + (str(index + 4801)).zfill(4) + '.xml'
        xml_shade_path_ori = 'F:\\anaconda\\project\\pattren_1\\train\\xml_shade\\train_' + (str(index + 1)).zfill(4) + '.xml'
        os.rename(xml_shade_path_ori, xml_shade_path)
        #update_xml(xml_shade_path, x1, y1, x2, y2)
        cv.imwrite(img_shade_path,  reSize1)
        cv.imshow('target', reSize1)
        cv.waitKey(1)
    print('sucessful')

'''
        filename = 'train_' + (str(index + 801)).zfill(4)
        x1, y1, x2, y2 = read_xml('./train/xml_shade/' + filename + '.xml')
        img = cv.imread('./train/image_shade/' + filename + '.jpg')
        cv.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
        cv.imshow('1', img)
        cv.waitKey(0)
 '''
