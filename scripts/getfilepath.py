import os
path = '/home/djw/PycharmProjects/Pattern_recognition/project3_to_student/fish_dataset/test/JPEGImages'
test_name = []
for file in os.listdir(path):
    test_name.append(file)

file1 = "/home/djw/PycharmProjects/tensorflow-yolov3/data/dataset/fish_test.txt"   ##train&val
#file1 = "/home/djw/PycharmProjects/tensorflow-yolov3/data/dataset/tiger_test.txt"  ##test
with open(file1,'r+') as f1:
    for line in test_name:
        # line = line.replace('.jpg','\n')  ##train&val
        line = path + os.path.sep + line + '\n'    ##test
        f1.writelines(line)

print("len_test_name:",len(test_name))