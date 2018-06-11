#-*- coding:utf-8 -*-
#功能：placeholder的用法
import tensorflow as tf

#参数给定类型 默认就是float32
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

#乘法运算
output=tf.mul(input1,input2)
with tf.Session() as sess:
    #注意placeholder与feed_dict是绑定的， 前面如果定义了placeholder就必须使用feed_dict喂给它值
    print sess.run(output,feed_dict={input1:[7],input2:[2.0]})
