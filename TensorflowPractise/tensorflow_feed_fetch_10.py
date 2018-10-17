# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/17 0017 下午 12:30
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: tensorflow中feed与fetch的用法
"""
import tensorflow as tf
import os

# 只显示warning 和error的信息
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 构建一个矩阵的乘法，但是矩阵数据在运行时给定（相当于定义一个方法）
m1=tf.placeholder(dtype=tf.float32,shape=[2,3],name='placeholder_m1')
#<class 'tensorflow.python.framework.ops.Tensor'>
print(type(m1))
m2=tf.placeholder(dtype=tf.float32,shape=[3,2],name='placeholder_m2')
m3=tf.matmul(m1,m2)

with tf.Session() as sess:
    result=sess.run(fetches=m3,feed_dict={
        m1:[[1,2,3],[4,5,6]],m2:[[1,2],[2,3],[3,4]]
    })
    print('result:{}'.format(result))
    
