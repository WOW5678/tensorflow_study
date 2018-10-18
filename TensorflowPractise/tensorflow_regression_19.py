# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/18 0018 下午 7:12
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 使用tensorflow实现一个简单的线性回归的案例
"""
import tensorflow as tf
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 只显示warning和error的信息
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#当绘图中含有中文时，必须加入这两行代码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

# 线性回归就是最简单的神经网络

# 1.构造数据
np.random.seed(1001)
N=100
#loc为均值，scale为方差
noise_x=np.random.normal(loc=0.0,scale=2,size=N)
x=np.linspace(0,6,N)

noise_y=np.random.normal(loc=0.0,scale=5,size=N)
y=14*x-7+noise_y

# 将x和y设置成矩阵
x.shape=(-1,1)
y.shape=(-1,1)
# print(x)

# 2.模型构建，y=wx+b
# 定义一个变量w和b
tf.set_random_seed(0)
w=tf.Variable(initial_value=tf.random_uniform(shape=[1],minval=-1.0,maxval=1.0,name='w'))
b=tf.Variable(initial_value=tf.zeros(shape=[1]),name='b')

# 3.得到一个预测值
y_hat=w*x+b

# 4.构建一个损失函数（以mse作为损失函数：预测值与实际值差的平方法做均值）
loss=tf.reduce_mean(tf.square(y-y_hat),name='loss')

# 5. 以随机梯度下降的方法优化损失函数
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05)
train=optimizer.minimize(loss)

# 6.执行阶段
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #进行多次训练
    for i in range(20):
        sess.run(train)
        # 输入每轮训练后的w b loss
        r_w,r_b,r_loss=sess.run([w,b,loss])
        print('w={},b={},loss={}'.format(r_w,r_b,r_loss))

# 7.画图
plt.figure(figsize=(12,6),facecolor='w')
# 为了画那条直线，需要产生很多模拟的数据
x_hat=np.linspace(x.min(),x.max(),num=50)
x_hat.shape=-1,1
y_hat2=r_w*x_hat+r_b

plt.plot(x,y,'ro',ms=5,label=u'实际值',zorder=2)
plt.plot(x_hat,y_hat2,color='b',lw=2,alpha=0.75,label='TF梯度下降，损失函数值=%.3f'%r_loss,zorder=2)
plt.xlabel(u'X',fontsize=16)
plt.ylabel(u'Y',fontsize=16)
plt.legend(loc='upper left')
plt.grid(True)
plt.title('Tensorflow实现线性回归')
plt.show()



