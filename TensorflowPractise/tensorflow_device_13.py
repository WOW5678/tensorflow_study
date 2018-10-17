# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/17 0017 下午 6:44
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 指定运算的设备
"""
import tensorflow as tf
import os

#只显示warning 和Error
with tf.device('/cpu:0'):
    # 这个代码块中定义的操作会在tf.device给定的设备中运行
    #有一些操作是不会再GPU上运行的（特别注意）
    #如果按照的tf cpu版本，则无法执行运行的设备

    a=tf.Variable(initial_value=[1,2,3],dtype=tf.int32,name='a')
    b=tf.constant(2,dtype=tf.int32,name='b')
    # 一个数组加上一个实数，则是给数组中每个元素加上实数
    c=tf.add(a,b,name='c')
    init_op=tf.global_variables_initializer()

with  tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(c))
