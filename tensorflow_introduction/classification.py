# -*- coding:utf-8 -*-
#功能：实现一个分类器
import tensorflow as tf
import numpy as np
from  tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

def addLayer(inputs,in_size,out_size,activation_function=None,):
    #定义随机的权重 规格为[in_size,out_size] 推荐不为0
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    #定义偏向，偏向是个列表 或者行向量 推荐不为0
    bias=tf.Variable(tf.zeros([1,out_size])+0.1)

    wx_plus_b=tf.matmul(inputs,Weights)+bias
    if activation_function==None:
        #为空即为线性激活函数
        output=wx_plus_b
    else:
        output=activation_function(wx_plus_b)
    return output

#计算预测的准确率  传入的参数为 输入和真实的标签
def compute_accuracy(v_xs,v_ys):
    global  prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

#定义palceholder
xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])
#增加输出层
prediction=addLayer(xs,784,10,activation_function=tf.nn.softmax)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

train_step=tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess=tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(100):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%10==0:
        print (compute_accuracy(mnist.test.images,mnist.test.labels))


