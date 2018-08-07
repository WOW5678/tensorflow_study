# -*- coding:utf-8 -*-
'''
function；使用minist 数据集实现 实现一个简单的mlp分类程序
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# 加载数据
mnist=input_data.read_data_sets('data\mnist',one_hot=True)

# define the classification model
x=tf.placeholder(tf.float32,[None,784])
#真实值
y=tf.placeholder(tf.float32,[None,10])
def set_weights(x,shape,activation_fun=None):
    w=tf.Variable(tf.random_normal(shape,stddev=0.01))
    b=tf.Variable(tf.zeros((1,shape[1])))
    w_plus_b=tf.matmul(x,w)+b
    if activation_fun:
        output=activation_fun(w_plus_b)
    else:
        output=w_plus_b
    return output

# Create model
h1=set_weights(x,[784,50],activation_fun=None)
h2=set_weights(h1,[50,20],activation_fun=None)
y_=set_weights(h2,[20,10],activation_fun=None)

correct_prediction=tf.equal(tf.argmax(y_,1),tf.argmax(y,1)) #计算预测值与真实值是否相等
#布尔值 转换成浮点数，并取平均值 得到的就是准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 定义损失函数和优化器
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_,labels=y))
# SGD优化目标
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 训练模型
#与常规会话不同的是，交互式会话会成为默认会话
sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
epochs=1000
for i in range(epochs):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    _,cross=sess.run([train_step,cross_entropy],feed_dict={x:batch_xs,y:batch_ys})
    if i%10==0 or i==epochs-1:
        print('epoch:{} loss:{}'.format(i,cross))
        print('accuracy:{}'.format(sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys})))
# 保存模型
saver=tf.train.Saver()
saverDir='checkpoints'
import os
if not os.path.exists(saverDir):
    os.makedirs(saverDir)
saver.save(sess=sess,save_path=os.path.join(saverDir,'final_model'))
sess.close()

