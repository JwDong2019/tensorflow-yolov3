import os
import shutil

pre_txt_path = '/home/djw/PycharmProjects/tensorflow-yolov3/mAP/predicted'
txt_name = []

predicted_file_path = "/home/djw/PycharmProjects/tensorflow-yolov3/mAP/predict_train_all.txt"
if os.path.exists(predicted_file_path): os.remove(predicted_file_path)
os.mknod(predicted_file_path)
for file in os.listdir(pre_txt_path):
    with open(pre_txt_path + os.path.sep + file) as f:
        for line in f.readlines():
            with open(predicted_file_path,'a') as f1:
                f1.writelines(line)