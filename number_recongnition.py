import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

import cv2
test = cv2.imread('test1.png')
a = np.zeros((28,28,1))
a[:,:,0] = test[:,:,1]

session = tf.Session()
saver = tf.train.import_meta_graph('model/number.meta')  # 先加载meta文件，具体到文件名
saver.restore(session, tf.train.latest_checkpoint('model'))  # 加载检查点文件checkpoint，具体到文件夹即可
graph = tf.get_default_graph()  # 绘制tensorflow图

xs = graph.get_tensor_by_name('input/x:0')  # 获取占位符xs
ys = graph.get_tensor_by_name('input/y:0')  # 获取占位符ys
output = graph.get_tensor_by_name('prediction/output:0')
result_array = session.run(output, feed_dict={xs:[a]})
print(np.argmax(result_array,1))