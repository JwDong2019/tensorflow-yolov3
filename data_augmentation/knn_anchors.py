import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import xml.etree.ElementTree as ET

LABELS = ['fish']

# train_image_folder = "/home/djw/Downloads/VOC/VOCdevkit/VOC2007/JPEGImages/"
# train_annot_folder = "/home/djw/Downloads/VOC/VOCdevkit/VOC2007/Annotations/"
train_image_folder = "./image/"
train_annot_folder = "./xml/"


def parse_annotation(ann_dir, img_dir, labels=[]):
    '''
    output:
    - Each element of the train_image is a dictionary containing the annoation infomation of an image.
    - seen_train_labels is the dictionary containing
            (key, value) = (the object class, the number of objects found in the images)
    '''
    all_imgs = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        if "xml" not in ann:
            continue
        img = {'object': []}

        # tree = ET.parse(ann_dir + ann)
        tree = ET.ElementTree(file= ann_dir + ann)

        for elem in tree.iter():
            #print(elem.tag)
            '''
            if 'filename' in elem.tag:
                path_to_image = img_dir + elem.text
                img['filename'] = path_to_image
                ## make sure that the image exists:
                if not os.path.exists(path_to_image):
                    assert False, "file does not exist!\n{}".format(path_to_image)
            '''
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:

                        obj['name'] = attr.text

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels


## Parse annotations
train_image, seen_train_labels = parse_annotation(train_annot_folder, train_image_folder, labels=LABELS)
# print("N train = {}".format(len(train_image)))
# print("train_image:", train_image)

# print('train_image[:2]:', train_image[:2])

# #####  show all kinds of objects number
# y_pos = np.arange(len(seen_train_labels))
# fig = plt.figure(figsize=(13,10))
# ax = fig.add_subplot(1,1,1)
# ax.barh(y_pos,list(seen_train_labels.values()))
# ax.set_yticks(y_pos)
# ax.set_yticklabels(list(seen_train_labels.keys()))
# ax.set_title("The total number of objects = {} in {} images".format(
#     np.sum(list(seen_train_labels.values())),len(train_image)
# ))
# plt.show()


#### K-means clustering  ###
#我们首先为K-means聚类准备要输入数据。 输入数据指的是ground truth bounding box的宽度和高度来作为特征。
# 考虑到在不同尺度下的场景中，每个boundingbox的尺寸不一。
# 因此，非常有必要来标准化边界框的宽度和高度与图像的宽度和高度。
wh = []
for anno in train_image:
    aw = float(anno['width'])  # width of the original image
    ah = float(anno['height']) # height of the original image
    for obj in anno["object"]:
        w = (obj["xmax"] - obj["xmin"])/aw # make the width range between [0,GRID_W)
        h = (obj["ymax"] - obj["ymin"])/ah # make the width range between [0,GRID_H)
        temp = [w,h]
        wh.append(temp)
wh = np.array(wh)
# print("clustering feature data is ready. shape = (N object, width and height) =  {}".format(wh.shape))
#
#
# ####  Visualize the clustering data  ###
# ####  先来看看归一化后的anchor尺寸分布情况   ####
# plt.figure(figsize=(10,10))
# plt.scatter(wh[:,0],wh[:,1],alpha=0.3)
# plt.title("Clusters",fontsize=20)
# plt.xlabel("normalized width",fontsize=20)
# plt.ylabel("normalized height",fontsize=20)
# plt.show()

def iou(box, clusters):
    '''
    :param box:      np.array of shape (2,) containing w and h
    :param clusters: np.array of shape (N cluster, 2)
    '''
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_

###The k-means clustering###
# K-means的聚类方法很简单，它主要包含两个步骤:
#
# 首先初始化类别数量和聚类中心:
#
# Step 1: 计算每个boundingbox与所有聚类中心的距离（1-iou)，选择最近的那个聚类中心作为它的类别
# Step 2: 使用每个类别簇的均值来作为下次迭代计算的类别中心
# 重复步骤1和2,直至每个类别的中心位置不再发生变化。

def kmeans(boxes, k, dist=np.median, seed=1):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))  ## N row x N cluster
    last_clusters = np.zeros((rows,))

    np.random.seed(seed)

    # initialize the cluster centers to be k items
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        # Step 1: allocate each item to the closest cluster centers
        for icluster in range(k):  # I made change to lars76's code here to make the code faster
            distances[:, icluster] = 1 - iou(clusters[icluster], boxes)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        # Step 2: calculate the cluster centers as mean (or median) of all the cases in the clusters.
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters, nearest_clusters, distances

###   The number of Clusters   ###
# 一般来说，anchor聚类的类别越多，那么yolo算法就越能在不同尺度下与真实框进行回归，但是这样也增加了很多计算量。
# (这对于一个号称 real-time 目标检测框架来说是极其尴尬的，因此作者也尽量减少boundingbox的数目)。
kmax = 11
dist = np.mean
results = {}

for k in range(2,kmax):
    clusters, nearest_clusters, distances = kmeans(wh,k,seed=2,dist=dist)
    WithinClusterMeanDist = np.mean(distances[np.arange(distances.shape[0]),nearest_clusters])
    result = {"clusters":             clusters,
              "nearest_clusters":     nearest_clusters,
              "distances":            distances,
              "WithinClusterMeanDist": WithinClusterMeanDist}
    # i = 0
    # for cluster in clusters:
    #     if i < 3:
    #         cluster = 8 * cluster
    #         clusters[i] = cluster
    #         i = i + 1
    #     elif i < 6:
    #         cluster = 16 * cluster
    #         clusters[i] = cluster
    #         i = i + 1
    #     elif i < 9:
    #         cluster = 32 * cluster
    #         clusters[i] = cluster
    #         i = i + 1
    # clusters = clusters.tolist()

    print("{:2.0f} clusters: mean IoU = {:5.4f}".format(k,1-result["WithinClusterMeanDist"]))
    print("clusters:", clusters)
    print("nearest_clusters:", nearest_clusters)
    results[k] = result


# 类别的数量越多，每个聚类簇的均值iou就越大，说明聚类簇里的boundingbox愈加紧贴在一起。
# 有时候很难决定类别的数目，这也是k-means的一大痛点！
# 在yolov2论文里设置了5个先验anchor，因此先来看看聚类数目从5到8的效果吧！

# ###   Visualization of k-means results   ###
def plot_cluster_result(plt, clusters, nearest_clusters, WithinClusterSumDist, wh, k):
    for icluster in np.unique(nearest_clusters):
        pick = nearest_clusters == icluster
        c = current_palette[icluster]
        plt.rc('font', size=8)
        plt.plot(wh[pick, 0], wh[pick, 1], "p",
                 color=c,
                 alpha=0.5, label="cluster = {}, N = {:6.0f}".format(icluster, np.sum(pick)))
        plt.text(clusters[icluster, 0],
                 clusters[icluster, 1],
                 "c{}".format(icluster),
                 fontsize=20, color="red")
        plt.title("Clusters=%d" % k)
        plt.xlabel("width")
        plt.ylabel("height")
    plt.legend(title="Mean IoU = {:5.4f}".format(WithinClusterSumDist))


import seaborn as sns

current_palette = list(sns.xkcd_rgb.values())

figsize = (15, 35)
count = 1
fig = plt.figure(figsize=figsize)
for k in range(2, kmax):
    result = results[k]
    clusters = result["clusters"]
    nearest_clusters = result["nearest_clusters"]
    WithinClusterSumDist = result["WithinClusterMeanDist"]

    # ax = fig.add_subplot(kmax / 2, 2, count)
    ax = fig.add_subplot(kmax / 3 , 3, count)
    plt.subplots_adjust(wspace=0.2, hspace=0.25)  # 调整子图间距

    plot_cluster_result(plt, clusters, nearest_clusters, 1 - WithinClusterSumDist, wh, k)
    count += 1
plt.show()