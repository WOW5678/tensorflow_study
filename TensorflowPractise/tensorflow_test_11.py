# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/17 0017 下午 12:50
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 有三个编程小练习，（1）累加器（2）更新变量维度（3）求解阶乘
"""
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#我的累加器(我的更简洁一些)
a=tf.Variable(initial_value=0,dtype=tf.int32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        result=sess.run(a)
        print('result:{}'.format(sess.run(tf.assign(ref=a,value=result+1))))

# 给定代码
i=tf.Variable(initial_value=0,dtype=tf.int32,name='i_var')
assign_op=tf.assign(ref=i,value=tf.add(i,tf.constant(1,dtype=tf.int32)),name='assign_op')

init_op=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for j in range(10):
        sess.run(assign_op)
        result=sess.run(i)
        print('result:{}'.format(result))


# 需求2：实现动态的更新变量的维度数目
# 定义一个不定形的变量
a_2=tf.Variable(initial_value=[],dtype=tf.float32,validate_shape=False)
#更改a的形状
b_2=tf.concat([a_2,[1,0,2.0]],axis=0)
assign_op_2=tf.assign(ref=a_2,value=b_2,validate_shape=False)
init_op_2=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op_2)
    for i in range(5):
        sess.run(assign_op_2)
        print(sess.run(a_2))

# 需求3：求解阶乘

# def jiecheng(n):
#     if n==1:
#         return 1
#     return n*(n-1)
# print(jiecheng(10))

a=tf.placeholder(dtype=tf.int32)
result=tf.Variable(initial_value=1,dtype=tf.int32,name='result')
init_op_4=tf.global_variables_initializer()
assign_op_4=tf.assign(ref=result,value=result*a)

with tf.Session() as sess:
    sess.run(init_op_4)
    for i in range(1,11):
        #sess.run(assign_op_4,feed_dict={a:i})
        print('assign:{}'.format(sess.run(assign_op_4,feed_dict={a:i})))
    print('result:{}'.format(sess.run(result)))


