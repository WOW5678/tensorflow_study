# -*- coding:utf-8 -*-
#功能：使用tensorflow保存训练好的模型
import tensorflow as tf
import numpy as np

#保存模型
#一定要指明类型,否则加载模型时会报错
# W=tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
# biaes=tf.Variable([[1,2,3]],dtype=tf.float32,name='biaes')
# init=tf.initialize_all_variables()
#
#
# saver=tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init)
#     saver_path=saver.save(sess,'mynet/save_net.ckpt')
#     print "save to path:" ,saver_path

#加载模型
#因为只保存了变量 所以神经网络的框架要重新进行定义要与 保存模型的shape 和dtype保持一致

w=tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name="weights")
b=tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biaes')
#
saver=tf.train.Saver()
with tf.Session() as sess:
    #restore函数重新加载训练好的模型
    saver.restore(sess,'mynet/save_net.ckpt')
    #权重和偏见
    print ('weights:',sess.run(w))
    print ('biaes:',sess.run(b))
