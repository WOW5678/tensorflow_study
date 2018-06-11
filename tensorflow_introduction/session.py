#-*- coding:utf-8 -*-
#功能：session的用法
import tensorflow as tf
import numpy as np

matrix1=tf.constant([[3,3]])
matrix2=tf.constant([[2],[2]])
#在numpy中矩阵的乘法位np.dot(m1,m2)
product=tf.matmul(matrix1,matrix2)

#method1
# sess=tf.Session()
# #每调用一次run就执行一个结构
# result=sess.run(product)
# print result
# sess.close()

#method2
with tf.Session() as sess:
    result2=sess.run(product)
    print (result2)