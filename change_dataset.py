import os
import random
import shutil
import skimage.io as io


def copyFile(fileDir, tarDir):
    pathDir = os.listdir(fileDir)
    for filename in pathDir:
        print(filename)

    print(len(pathDir))  # 打印图片数量
    file = "/home/kb539/Downloads/tiger_dataset/train/ImageSets/Main/val.txt"
    val_pic = []
    with open(file) as f:
        for line in f.readlines():
            line = line.replace('\n','.jpg')
            val_pic.append(line)

    for name in val_pic:
        #shutil.copyfile(fileDir + name, tarDir + name)
        shutil.move(fileDir + name, tarDir)


if __name__ == '__main__':
    fileDir = "/home/kb539/Downloads/tiger_dataset/train/JPEGImages/"  # 填写要读取图片文件夹的路径
    tarDir = "/home/kb539/Downloads/tiger_dataset/train/train/"  # 填写保存随机读取图片文件夹的路径
    str = 'fileDir*.jpg'  # fileDir的路径+*.jpg表示文件下的所有jpg图片
    copyFile(fileDir, tarDir)