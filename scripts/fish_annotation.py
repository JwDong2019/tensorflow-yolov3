import os
import argparse
import xml.etree.ElementTree as ET

def convert_tiger_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):

    classes = ['fish']
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    with open(anno_path, 'w') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            print(annotation)
            f.write(annotation + "\n")
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/djw/PycharmProjects/Pattern_recognition/project3_to_student/fish_dataset/")
    parser.add_argument("--train_annotation", default="../data/dataset/fish_train.txt")
    parser.add_argument("--val_annotation", default="../data/dataset/fish_val.txt")
    #parser.add_argument("--test_annotation",  default="../data/dataset/tiger_test.txt")
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)
    #if os.path.exists(flags.test_annotation):os.remove(flags.test_annotation)

    num1 = convert_tiger_annotation(os.path.join(flags.data_path, 'train'), 'train', flags.train_annotation, False)
    num2 = convert_tiger_annotation(os.path.join(flags.data_path, 'val'), 'val', flags.val_annotation, False)
    #num3 = convert_tiger_annotation(os.path.join(flags.data_path, 'test'),  'test', flags.test_annotation, False)
    #print('=> The number of image for train is: %d\tThe number of image for val is: %d\tThe number of image for test is:%d' %(num1, num2, num3))
    print('=> The number of image for train is: %d\tThe number of image for val is: %d' %(num1, num2))


