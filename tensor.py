from skimage import io, transform
import glob
import os
import tensorflow as tf
import numpy as np
import time

# 数据集地址
path = 'J:/deep learning/project/kk/'
path_2 = 'J:/deep learning/project/kk_valid/'
# 模型保存地址
model_path = 'J:/deep learning/project/model.ckpt'

# 将所有的图片resize成100*100
w = 100
h = 100
c = 4


# 读取图片
def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        print(folder)
        for im in glob.glob(folder + '/*.jpg'):
            #print('reading the images:%s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            labels.append(idx)
    print(len(imgs))
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


data, label = read_img(path)
data_2,label_2 = read_img(path_2)

# 打乱顺序
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

# 将所有数据分为训练集和验证集
ratio = 0.8
s = np.int(num_example * ratio)
x_train = data
y_train = label
x_val = data_2
y_val = label_2

# -----------------构建网络----------------------
# 占位符
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [5, 5, 4, 32],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight", [5, 5, 32, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight", [4, 4, 64, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer7-conv4"):
        conv4_weights = tf.get_variable("weight", [4, 4, 128, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.name_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        nodes = 6 * 6 * 128
        reshaped = tf.reshape(pool4, [-1, nodes])

    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [512, 5],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [5], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit


# ---------------------------网络结束---------------------------
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = inference(x, False, regularizer)

# (小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1, dtype=tf.float32)
logits_eval = tf.multiply(logits, b, name='logits_eval')

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# 训练和测试数据，可将n_epoch设置更大一些

n_epoch = 5
batch_size = 64
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("D://TensorBoard//test",sess.graph)
for epoch in range(n_epoch):
    start_time = time.time()
    print("\n")
    print("epoch: %i"% epoch)
    # training7huu
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err;
        train_acc += ac;
        n_batch += 1
    print("   train loss: %f" % (np.sum(train_loss) / n_batch))
    print("   train acc: %f" % (np.sum(train_acc) / n_batch))

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err;
        val_acc += ac;
        n_batch += 1
    print("   validation loss: %f" % (np.sum(val_loss) / n_batch))
    print("   validation acc: %f" % (np.sum(val_acc) / n_batch))
writer.close()
saver.save(sess, model_path)
sess.close()

# # -*- coding: utf-8 -*-
#
# from skimage import io, transform
# import glob
# import os
# import tensorflow as tf
# import numpy as np
# import time
#
# path = 'e:/flower/'
#
# # 将所有的图片resize成100*100
# w = 100
# h = 100
# c = 3
#
#
# # 读取图片
# def read_img(path):
#     cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
#     imgs = []
#     labels = []
#     for idx, folder in enumerate(cate):
#         for im in glob.glob(folder + '/*.jpg'):
#             print('reading the images:%s' % (im))
#             img = io.imread(im)
#             img = transform.resize(img, (w, h))
#             imgs.append(img)
#             labels.append(idx)
#     return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)
#
#
# data, label = read_img(path)
#
# # 打乱顺序
# num_example = data.shape[0]
# arr = np.arange(num_example)
# np.random.shuffle(arr)
# data = data[arr]
# label = label[arr]
#
# # 将所有数据分为训练集和验证集
# ratio = 0.8
# s = np.int(num_example * ratio)
# x_train = data[:s]
# y_train = label[:s]
# x_val = data[s:]
# y_val = label[s:]
#
# # -----------------构建网络----------------------
# # 占位符
# x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
# y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')
#
# # 第一个卷积层（100——>50)
# conv1 = tf.layers.conv2d(
#     inputs=x,
#     filters=32,
#     kernel_size=[5, 5],
#     padding="same",
#     activation=tf.nn.relu,
#     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
# pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
#
# # 第二个卷积层(50->25)
# conv2 = tf.layers.conv2d(
#     inputs=pool1,
#     filters=64,
#     kernel_size=[5, 5],
#     padding="same",
#     activation=tf.nn.relu,
#     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
# pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#
# # 第三个卷积层(25->12)
# conv3 = tf.layers.conv2d(
#     inputs=pool2,
#     filters=128,
#     kernel_size=[3, 3],
#     padding="same",
#     activation=tf.nn.relu,
#     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
# pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
#
# # 第四个卷积层(12->6)
# conv4 = tf.layers.conv2d(
#     inputs=pool3,
#     filters=128,
#     kernel_size=[3, 3],
#     padding="same",
#     activation=tf.nn.relu,
#     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
# pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
#
# re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])
#
# # 全连接层
# dense1 = tf.layers.dense(inputs=re1,
#                          units=1024,
#                          activation=tf.nn.relu,
#                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
#                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
# dense2 = tf.layers.dense(inputs=dense1,
#                          units=512,
#                          activation=tf.nn.relu,
#                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
#                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
# logits = tf.layers.dense(inputs=dense2,
#                          units=5,
#                          activation=None,
#                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
#                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
# # ---------------------------网络结束---------------------------
#
# loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
# train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
# acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#
# # 定义一个函数，按批次取数据
# def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
#     assert len(inputs) == len(targets)
#     if shuffle:
#         indices = np.arange(len(inputs))
#         np.random.shuffle(indices)
#     for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
#         if shuffle:
#             excerpt = indices[start_idx:start_idx + batch_size]
#         else:
#             excerpt = slice(start_idx, start_idx + batch_size)
#         yield inputs[excerpt], targets[excerpt]
#
#
# # 训练和测试数据，可将n_epoch设置更大一些
#
# n_epoch = 10
# batch_size = 64
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# for epoch in range(n_epoch):
#     start_time = time.time()
#
#     # training
#     train_loss, train_acc, n_batch = 0, 0, 0
#     for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
#         _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
#         train_loss += err;
#         train_acc += ac;
#         n_batch += 1
#     print("   train loss: %f" % (train_loss / n_batch))
#     print("   train acc: %f" % (train_acc / n_batch))
#
#     # validation
#     val_loss, val_acc, n_batch = 0, 0, 0
#     for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
#         err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
#         val_loss += err;
#         val_acc += ac;
#         n_batch += 1
#     print("   validation loss: %f" % (val_loss / n_batch))
#     print("   validation acc: %f" % (val_acc / n_batch))
#
# sess.close()