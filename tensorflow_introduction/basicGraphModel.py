# -*- coding:utf-8 -*-
#功能：介绍tensorflow的基础图
import tensorflow as tf
import  matplotlib.pyplot as plt
#创建图
num=32
x=tf.linspace(-3.0,3.0,num)

#构建一个高斯分布
sigma = 1.0
mean = 0.0
# 高斯分布函数
z = (1.0 / (sigma * tf.sqrt(2 * 3.14))) * tf.exp(-(tf.pow(x - mean, 2.0) / (2.0 * tf.pow(sigma, 2))))
assert z.graph is tf.get_default_graph()

#启动图
with tf.Session() as sess:
    result=sess.run(x)
    print (result)
    z=sess.run(z)

# #启动图的另外一种方式
# sess=tf.InteractiveSession()
# #直接对tnsor变量调用eval()函数就可以得到tensor的具体数值
# print x.eval()


#使用matplotlib画出高斯分布曲线
    plt.plot(z)
    plt.show()

