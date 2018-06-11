# -*- coding:utf-8 -*-
#功能：实现tensorflow结构的可视化

import tensorflow as tf
import numpy as np

def addLayer(inputs,in_size,out_size,activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weight'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]))
        with tf.name_scope('biaes'):
            bias=tf.Variable(tf.zeros([1,out_size])+0.1)
        with tf.name_scope('wx_plus_bias'):
            wx_plus_b=tf.matmul(inputs,Weights)+bias
        if activation_function==None:
            output=wx_plus_b
        else:
            output=activation_function(wx_plus_b)
        return output

#创建数据
x_data=np.linspace(-1,1,300)[:,np.newaxis]
#噪音
noise=np.random.normal(0,0.1,x_data.shape)
y_data=np.square(x_data)-0.5+noise

with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name="x_input")
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')

#神经网络的结构
l1=addLayer(xs,1,10,tf.nn.relu)
prediction=addLayer(l1,10,1,None)

with tf.name_scope('loss'):
    #定义误差函数
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
with tf.name_scope('train'):
    #定义训练方式
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)


#初始变量
sess=tf.Session()
writer=tf.train.SummaryWriter('logs/',sess.graph)
sess.run(tf.initialize_all_variables())

#训练过程
for i in range(100):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%10 ==0:
        print (sess.run(loss,feed_dict={xs:x_data,ys:y_data}))