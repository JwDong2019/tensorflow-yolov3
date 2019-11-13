#coding:utf-8
import re
import os

'''
file1 = "/home/kb539/Downloads/tiger_dataset/train/ImageSets/Main/train.txt"
val_pic = []
with open(file1) as f1:
    for line in f1.readlines():
        line = line.replace('\n', '.jpg')
        #line = line.strip("\n")
        val_pic.append(line)
    print('val_pic:', val_pic)
    print('len_val_pic:', len(val_pic))
    '''

path = '/home/djw/Downloads/tiger_dataset/test/JPEGImages/'
test_name = []
for file in os.listdir(path):
    test_name.append(file)

file2 = "/home/kb539/PycharmProjects/djw/tensorflow-yolov3/data/dataset/tiger_test.txt"
#file3 = "/home/kb539/PycharmProjects/djw/tensorflow-yolov3/data_tiger/val.txt"
#file3 = "/home/kb539/PycharmProjects/djw/tensorflow-yolov3/data_tiger/test.txt"
file3 = "/home/kb539/PycharmProjects/djw/tensorflow-yolov3/data/dataset/tiger_train.txt"
with open(file2) as f2:
    with open(file3,'w') as f3:
        for line in f2.readlines():
            for i in test_name:
                n = re.findall(i, line)
                if n:
                    f3.writelines(line)