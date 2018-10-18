# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/18 0018 上午 9:22
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 了解tf变量作用域的用法和使用
"""
import tensorflow as tf
import os
#只显示Warning 和 error的信息
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#方式1 我们之前的方式，变量数量会比较多的情况
# 会发现w1和w2的格式格式几乎完全一样，但是要写两份
def my_fun(x):
    #w1 b1都只有一个元素
    w1=tf.Variable(initial_value=tf.random_normal(shape=[2]))[0]
    b1=tf.Variable(initial_value=tf.random_normal(shape=[1]))[0]
    r1=w1*x+b1

    w2=tf.Variable(initial_value=tf.random_normal(shape=[2]))[0]
    b2=tf.Variable(initial_value=tf.random_normal(shape=[1]))[0]
    r2=w2*r1+b2
    return w1,b1,r1,w2,b2,r2

# 方式二 共享变量名，但使用作用域区分
def my_fun2(x):
    # initializer初始化器
    w=tf.get_variable(name='w',shape=[2],initializer=tf.random_normal_initializer())[0]
    b=tf.get_variable(name='b',shape=[1],initializer=tf.random_normal_initializer())[0]
    r=w*x+b
    return w,b,r

# def fun2(x):
#     r1=my_fun2(x)
#     r2=my_fun2(r1[2])
#     return r1,r2

def fun2(x):
    #当reuse的值为tf.AUTO_REUSE的时候，如果变量存在就重用变量，否则创建变量并返回
    with tf.variable_scope(name_or_scope='op1',reuse=tf.AUTO_REUSE):
        r1=my_fun2(x)
    with tf.variable_scope(name_or_scope='op2',reuse=tf.AUTO_REUSE):
        r2=my_fun2(r1[2])
    return r2
#调用函数
x=tf.constant(3.0,name='x')
#result=my_fun(x)
result=fun2(x)
init_op=tf.global_variables_initializer()

# 再次调用这个函数
x2=tf.constant(4.0,name='x2')
result2=fun2(x2)

with tf.Session()as sess:
    sess.run(init_op)
    print('result:{}'.format(sess.run(result)))
    print('result2:{}'.format(sess.run(result2)))



