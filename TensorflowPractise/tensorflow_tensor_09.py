# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/17 0017 下午 12:12
 @Author  : Shanshan Wang
 @Version : Python3.5
 @function: tensorflow张量、变量的区别于使用
"""
import tensorflow as tf
import os

#只显示warning 和error
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 1.定义一个变量，必须给定初始值
a=tf.Variable(initial_value=3.0,dtype=tf.float32)
#<class 'tensorflow.python.ops.variables.Variable'>
print(type(a))
aa=2
#<class 'int'>
print(type(aa))
# 2.定义一个张量
b=tf.constant(value=2.0,dtype=tf.float32)
#<class 'tensorflow.python.framework.ops.Tensor'>
print(type(b))
c=tf.add(a,b)

# 3.变量的初始化（推荐：使用全局所有变量的初始化API）
#相当于在图中加入一个初始化全局所有变量的操作
#init_op=tf.initialize_all_variables()
init_op=tf.global_variables_initializer()
#<class 'tensorflow.python.framework.ops.Operation'>
print(type(init_op))
#Tensor("Variable/initial_value:0", shape=(), dtype=float32)
print(a.initial_value)


# 4. 图的执行阶段
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    # 运行init_op进行变量的初始化，一定刚要方法所有操作之前
    sess.run(init_op)
    #另外一种初始化操作,但是要求明确给定当前代码块对应的默认Session是哪个？底层使用默认session（tf.get_default_session()）
    # init_op.run()
    #获取操作的结果
    print('result:{}'.format(sess.run(c)))
    print('result:{}'.format(c.eval()))
