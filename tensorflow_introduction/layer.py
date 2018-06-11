# -*- coding:utf-8 -*-
#功能：添加神经网络的层  并且可视化这个参数调整的过程
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def addLayer(inputs,in_size,out_size,activation_function=None):
    #定义随机的权重 规格为[in_size,out_size] 推荐不为0
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    #定义偏向，偏向是个列表 或者行向量 推荐不为0
    bias=tf.Variable(tf.zeros([1,out_size])+0.1)

    wx_plus_b=tf.matmul(inputs,Weights)+bias
    if activation_function==None:
        #为空即为线性激活函数
        output=wx_plus_b
    else:
        output=activation_function(wx_plus_b)
    return output

#创建数据
#linespace(-1,1,300)是之在-1,1之间生成300个点
# 再加上[:,tf.newaxis]是指增加agiel维度 即变成了300行1列的样本点
x_data=np.linspace(-1,1,300)[:,np.newaxis]
#print x_data
#定义y=x^2+0.5
#为了更切合实际 加入一些噪音 服从（0,0.05）的正太分布
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise



#None表示 给多少个例子都可以
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])
#神经网络的结构（1,10,1）
l1=addLayer(xs,1,10,activation_function=tf.nn.relu)
prediction=addLayer(l1,10,1,activation_function=None)

#定义平方误差
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
#定义训练方式
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始变量
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

#做图
#先尝试画出真实数据
figure=plt.figure()
ax=figure.add_subplot(1,1,1)
#scatter散点图
ax.scatter(x_data,y_data)
#这句代码的功能是不会暂停整个程序 或者 plt.show(block=False)也可以完成这个功能
plt.ion()
plt.show()

for i in range(100):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%10==0:
        try:
            #每次重新调整了参数后移除上次画的线条
            #因为第一次画图时没有lInes[0] 所以加上异常处理
            ax.lines.remove(lines[0])
        except:
            pass
        #print sess.run(loss,feed_dict={xs:x_data,ys:y_data})
        prediction_value=sess.run(prediction,feed_dict={xs:x_data,ys:y_data})
        #plot连线图
        lines=ax.plot(x_data,prediction_value,'r-',lw=5)
        #暂停1秒钟
        plt.pause(1)
        plt.show()


