# import libs
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import datasets
mnist = fetch_openml('mnist_784')

# define input and output
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None,28,28,1],name = 'x') # input
    y = tf.placeholder(tf.float32,[None,10],name = 'y') # output


# a module
def get_weight(shape):
    w_init = tf.random.truncated_normal(shape=shape, mean=0, stddev=0.1)
    b_init = tf.random.truncated_normal(shape=[shape[-1]], mean=0, stddev=0.1)

    w = tf.Variable(initial_value=w_init)
    b = tf.Variable(initial_value=b_init)
    return w, b

# 01
# define train variable
w1,b1 = get_weight([5,5,1,6])
# define cacu
o1 = tf.nn.conv2d(input = x,filter=w1,strides=[1,1,1,1],padding='SAME')
o1 = tf.nn.bias_add(o1,b1)
o1 = tf.nn.relu(o1)
o1 = tf.nn.avg_pool(value=o1, ksize=(1,2,2,1), strides=[1,2,2,1],padding='SAME')

# 02
# define train variable
w2,b2 = get_weight([5,5,6,16])
# define cacu
o2 = tf.nn.conv2d(input = o1,filter=w2,strides=[1,1,1,1],padding='VALID')
o2 = tf.nn.bias_add(o2,b2)
o2 = tf.nn.relu(o2)
o2 = tf.nn.avg_pool(value=o2, ksize=(1,2,2,1), strides=[1,2,2,1],padding='SAME')

# 03
# define train variable
w3,b3 = get_weight([5,5,16,120])
# define cacu
o3 = tf.nn.conv2d(input = o2,filter=w3,strides=[1,1,1,1],padding='VALID')
o3 = tf.nn.bias_add(o3,b3)
o3 = tf.nn.relu(o3)
o3 = tf.nn.avg_pool(value=o3, ksize=(1,2,2,1), strides=[1,2,2,1],padding='SAME')

# 2d->1d
o3 = tf.reshape(o3,[-1,120])
w4, b4 = get_weight([120,84])
o4 = tf.nn.relu(tf.matmul(o3,w4)+b4)
o4 = tf.nn.dropout(o4,0.75)


w5, b5 = get_weight([84,10])
with tf.name_scope('prediction'):
    o5 = tf.nn.softmax(tf.matmul(o4,w5)+b5,name='output')
y_ = o5

#define loss function
loss = tf.losses.sigmoid_cross_entropy(y,y_)
optimzer = tf.train.AdamOptimizer(0.0001)
trainer = optimzer.minimize(loss)

# load data

# load label
result  = mnist.target
result = result[:20000]

labels = np.zeros(  (len(result),10),  dtype = np.int)
for i in range(len(result)):
    lb = int(result[i])
    labels[i][lb] = 1


#load image
data = np.zeros( (len(result),28,28,1),np.float32)

mnist_image = mnist.data

# show number image
#plt.imshow(mnist_image[1].reshape(28,28),cmap = 'gray')
#plt.show()
for i in range(len(result)):
    data[i,:,:,0] = mnist_image[i].reshape(28,28)
print("data loaded")

# caculate accuracy rate
with tf.name_scope('accuracy'):
    correct = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float') ,name='accuracy')

# train
session = tf.Session()
global_v = tf.global_variables()
op_init = tf.initializers.variables(global_v)
session.run(op_init)

TIMES = 500
batch_size  = 100
batch = len(data) // batch_size
correct_rates = []
for t in range(TIMES):
    loss_result = 0.0
    for idx in range(batch):
        _, loss_result = session.run(
            [trainer,loss],
            feed_dict={
                x:data[idx*batch_size:(idx+1)*batch_size],
                y:labels[idx*batch_size:(idx+1)*batch_size]
            }
        )
    correct_rate = session.run(accuracy,feed_dict={x:data,y:labels})
   # print("correct rate: %5.2f%% loss:%f" % (correct_rate*100.0, loss_result))
    correct_rates.append(correct_rate)
print("train finshed")

# figure train process
figure = plt.figure(figsize=(8,4))
ax = figure.add_axes([0.1,0.1,0.8,0.8])
ax.plot(range(len(correct_rates)),correct_rates,
       color = (0,0,1,1),marker = '.',label = 'correct rate',
        markerfacecolor=(1,0,0,1),markeredgecolor = (1,0,0,1),markersize = 3
       )
ax.set_xbound(lower=-1,upper = len(correct_rates))
ax.set_ybound(lower=0,upper=1)
plt.legend()
plt.show()

import cv2
test = cv2.imread('./6.png')

a = np.zeros((28,28,1))
a[:,:,0] = test[:,:,1]

plt.imshow(a,cmap = 'gray')

p_result = tf.argmax(y_,1)
r = session.run(p_result,feed_dict={x:[a]})
print(r)

saver = tf.train.Saver()
saver.save(session, './model/number')
