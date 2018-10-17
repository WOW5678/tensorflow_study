# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/17 0017 下午 12:50
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: tensorflow 控制依赖的使用
"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


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

with tf.control_dependencies([assign_op_4]):
    # 如果需要代码块中的内容，必须执行control_dependencies 中给定的操作
    # 或tensor
    result=tf.Print(result,data=[result,result.read_value()],message='result:')

with tf.Session() as sess:
    sess.run(init_op_4)
    for i in range(1,7):
        #sess.run(assign_op_4,feed_dict={a:i})
        r=sess.run(result,feed_dict={a:i})
    print('result:{}'.format(r))


