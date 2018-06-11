# -*- coding:utf-8 -*-
#功能：使用卷积神经网络实现对手写数字的识别

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
#from libs.utils import *
import matplotlib.pyplot as plt

#新建权重函数和偏置函数
def weight_variable(shape):
    initial=tf.random_normal(shape,mean=0.0,stddev=0.01)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.random_normal(shape,mean=0.0,stddev=0.01)
    return tf.Variable(initial)

#新建输入输出有关的占位符
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

#构造两层卷积神经网络
#把数据维度变为4-D,即（batch_size,width,height,channel）
#bacth_size=-1的意思是这个维度只用总维度除以width*height*channel
x_tensor=tf.reshape(x,[-1,28,28,1])

#设置感知区域的大小
filter_size=3
n_filter_1=16
w_conv1=weight_variable([filter_size,filter_size,1,n_filter_1])
b_conv1=bias_variable([n_filter_1])

#第一次卷积后的结果
h_conv1=tf.nn.elu(tf.nn.conv2d(input=x_tensor,filter=w_conv1,strides=[1,2,2,1],padding='SAME')+b_conv1)

#第二个卷积核的参数
n_filter_2=16
w_conv2=weight_variable([filter_size,filter_size,n_filter_1,n_filter_2])
b_conv2=bias_variable([n_filter_2])

#第二个卷积之后的结果
h_conv2=tf.nn.elu(tf.nn.conv2d(input=h_conv1,filter=w_conv2,strides=[1,2,2,1],padding='SAME')+b_conv2)

#添加全连接隐层
#由卷积层过度到隐含层，需要对卷积层的输出做一个维度变换
h_conv2_flat=tf.reshape(h_conv2,[-1,7*7*n_filter_2])

#创造一个全连接层，隐含神经元的个数为1024
n_fc=1024
w_fc1=weight_variable([7*7*n_filter_2,n_fc])
b_fc1=bias_variable([n_fc])
#全连接之后的输出层
h_fc1=tf.nn.elu(tf.matmul(h_conv2_flat,w_fc1)+b_fc1)

#添加dropout 防止过拟合
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#添加softmax层
w_fc2=weight_variable([n_fc,10])
b_fc2=bias_variable([10])
y_pred=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

#获取目标函数
cross_entropy=-tf.reduce_sum(y*tf.log(y_pred))
optimizer=tf.train.AdamOptimizer().minimize(cross_entropy)

#计算分类准确率
correct_prediction=tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))

#新建会话 并进行mini_batch训练
sess=tf.Session()
sess.run(tf.initialize_all_variables())

#使用mini_batch来训练
batch_size=100
n_epoch=5
for epoch_i in range(n_epoch):
    for batch_i in range(mnist.train.num_examples//batch_size):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})

    #对所有mini_batch执行完一轮以后 打印结果
    print (sess.run(accuracy,feed_dict={x:mnist.validation.images,y:mnist.validation.labels,keep_prob:0.5}))
