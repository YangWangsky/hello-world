import tensorflow as tf
import numpy as np
import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差,这个函数产生正太分布.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积，1步长（stride size）、0边距（padding size）
#tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
#input：指需要做卷积的输入图像，[batch, in_height, in_width, in_channels],[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
#filter：相当于CNN中的卷积核，[filter_height, filter_width, in_channels, out_channels],[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
#strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
#padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
#SAME为补零步长不够的边界，VALLD则抛弃（SAME往往会导致卷积前后尺寸不变）

#use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
#结果返回一个Tensor，这个输出，就是我们常说的feature map
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



#池化，2*2的max pooling
#max_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
#value: 一个4D张量，格式为[batch, height, width, channels]，与conv2d中input格式一样
#ksize: 长为4的list,表示池化窗口的尺寸
#strides: 池化窗口的滑动值，与conv2d中的一样
#padding: 与conv2d中用法一样。

#http://www.myexception.cn/other/1815412.html 池化的size一般与步长相同，一般每个池化窗口都是不重复的
#那么卷积结果经过池化以后的结果，其尺寸应该是？*14*14*32  28/2=14
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])

#将x变为4d向量，第二维和第三维为图片的宽、高，第四为图片颜色通道数，灰度图为1
x_image = tf.reshape(x, [-1,28,28,1])

#第一层卷积
#每张图片的像素为28*28=784
#卷积在每个5x5的patch中算出32个特征，即32个卷积核，28*28*32(padding为SAME的原因)
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#卷积加上偏置后，进行relu(max（0，x）)，再池化
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#第二层卷积
#第二次卷积前图像尺寸为14*14*32，第二次卷积后为14*14*64（对32维度取平均值）
#池化后，输出图像尺寸为7*7*64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#第三层，全连接层
#全连接层有1024个神经元
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#drop out
keep_prob = tf.placeholder(tf.float32) # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#第四层  输出层 softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)#预测标签


y_ = tf.placeholder(tf.float32, [None, 10])  #真实标签
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.initialize_all_variables()) # 变量初始化
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],  y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
