import cv2
import matplotlib.pyplot as plt

img = cv2.imread("/home/djw/PycharmProjects/Pattern_recognition/project3_to_student/fish_dataset/train/JPEGImages/train_0001.jpg")
cv2.rectangle(img,(174,235),(287,312),(0, 255, 0), 3)
plt.imshow(img)
plt.show()