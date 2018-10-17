# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/17 0017 上午 9:26
 @Author  : Shanshan Wang
 @Version : Python3.5
"""
import os
import tensorflow as tf
# 默认的显示等级，会显示所有的信息
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
# 只显示warning 和error
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#只显示error
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

#定义常量矩阵ab
a=tf.constant([[1,2],[3,4]],dtype=tf.int32)
#<class 'tensorflow.python.framework.ops.Tensor'>
print(type(a))
b=tf.constant([5,6,7,8],dtype=tf.int32,shape=[2,2])

#以a b 作为输入进行矩阵的乘法操作
c=tf.matmul(a,b)
#<class 'tensorflow.python.framework.ops.Tensor'>
print(type(c))
# True
print('变量a是否在默认图中：{}'.format(a.graph is tf.get_default_graph()))

# 使用新构建的图，而不是默认图
graph1=tf.Graph()
with graph1.as_default():
    d=tf.constant(5.0,name='d')
    # True
    print('变量d是否在新图graph1中：{}'.format(d.graph is graph1))

# False
print('变量d是否在默认图中：{}'.format(d.graph is tf.get_default_graph()))

graph2=tf.Graph()
with graph2.as_default():
    e=tf.constant(3.0,name='e')
    # True
    print('变量e是否在新图graph2中：{}'.format(e.graph is graph2))
# 注意，不同使用两个图中的变量进行操作
# f=tf.add(d,e)

# 以c 和a 作为输入，进行矩阵的相加操作
g=tf.add(a,c,name='add')
print(g)

# 增加的操作，复杂点
h=tf.subtract(b,a,name='b-a')
i=tf.matmul(h,c,name='h_cheng_c')
j=tf.add(g,i,name='g_jia_i')

# 会话的创建、启动、关闭（默认情况下，创建的session属于默认图）
#sess=tf.Session(graph=tf.get_default_graph())
sess=tf.Session()
print(sess)

# 调用sess的run 方法执行矩阵的乘法，得到c的结果值
# 不需要考虑图中间的运算，在运行的时候只需要关注最终结果对应的对象以及所需要的输入数据值
# 会自动的根据图中的依赖关系触发所有相关的op操作的执行

# 如果op之间没有依赖关系，tensorflow底层会自定的并行执行op
result=sess.run(j)
print('type:{},value:{}'.format(type(result),result))

# 如果还需要得到c的结果
result2=sess.run(c)
print(result2)
# 如果传递的是fetches是一个列表（顺序没有关系），name返回值是一个list集合
result3=sess.run([j,c])
print('type:{},value:{}'.format(type(result3),result3))

#会话的关闭
sess.close()

# 当一个会话关闭后，不能使用了，所以以下是错误的
#RuntimeError: Attempted to use a closed Session.
# result4=sess.run(c)
# print(result4)

with tf.Session() as sess2:
    print(sess2)
    # 获取张量c的结果，通过session的run方法
    print('sess2 run:\n{}'.format(sess2.run(c)))
    #通过张量的eval方法同session的run方法是一样的
    print('c eval:\n{}'.format(c.eval()))

# 交互式会话的创建
sess3=tf.InteractiveSession()
print(j.eval())