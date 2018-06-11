# -*- coding:utf-8 -*-
#功能：实现一个例子
import tensorflow as tf
import numpy as np

#创建一个数据集
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

#创建tensorflow结构
Weight=tf.Variable(tf.random_uniform([1],-1.0,1.0))
bias=tf.Variable(tf.zeros([1]))

y=Weight*x_data+bias

loss=tf.reduce_mean(tf.square(y-y_data))
#0.5为学习效率
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

#初始化所有变量
init=tf.initialize_all_variables()
#创建tensorflow结构

sess=tf.Session()
sess.run(init)
#开始训练
for step in range(200):
    sess.run(train)
    if step%20==0:
        #每隔20步输出一次训练出的参数
        print (step,sess.run(Weight),sess.run(bias))

