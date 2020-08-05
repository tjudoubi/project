from skimage import io, transform
import tensorflow as tf
import numpy as np
import glob
import os
from xlwt import *

path1 = "J:/deep learning/project/flower/Fake/rich_zhang_ILSVRC2012_val_00034328.png"
path2 = "J:/deep learning/project/kk/false/rich_zhang_ILSVRC2012_val_00025275.jpg"
path3 = "E:/data/datasets/flower_photos/roses/394990940_7af082cf8d_n.jpg"
path4 = "E:/data/datasets/flower_photos/sunflowers/6953297_8576bf4ea3.jpg"
path5 = "E:/data/datasets/flower_photos/tulips/10791227_7168491604.jpg"

file = Workbook(encoding = 'utf-8')
#指定file以utf-8的格式打开
table = file.add_sheet('data')
#指定打开的文件名


flower_dict = {0: 'false', 1: 'true'}

w = 100
h = 100
c = 4


def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img, (w, h,c))
    return np.asarray(img)



with tf.Session() as sess:
    data = []
    name = []
    folder = "J:/deep learning/project/va"
    for root, dirs, files in os.walk(folder):
        name = files
    print(name)
    for im in glob.glob(folder + '/*.JPEG'):
        # print(im)
        data1 = read_one_image(im)
        data.append(data1)
    saver = tf.train.import_meta_graph('J:/deep learning/project/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('J:/deep learning/project/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x: data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits, feed_dict)

    # 打印出预测矩阵
    print(classification_result)
    # 打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result, 1).eval())
    # 根据索引通过字典对应花的分类
    output = []
    output = tf.argmax(classification_result, 1).eval()
    for i in range(len(output)):
        for j in range(2):
            if j == 0:
                table.write(i, j, name[i])
            else:
                table.write(i, j, flower_dict[output[i]])

    file.save('data10.xlsx')