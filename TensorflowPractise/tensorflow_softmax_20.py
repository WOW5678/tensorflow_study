# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/19 0019 上午 9:07
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function:利用tensorflow实现softmax分类问题
"""
import  tensorflow as tf
import os
import numpy as np

# 只显示warning 和 error信息
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# tensorflow 实现softmax分类问题案例

# 1.构造数据
N=20
np.random.seed(1001)
x_data=np.random.normal(loc=0,scale=2,size=[N,2])
y_data=np.dot(x_data,[[5],[-3]])
print('y_data:',y_data)
#二值化
y_data[y_data>0]=1
y_data[y_data<0]=0
print('y_data:',y_data)
y_data1=1-y_data
print('y_data:',y_data)
# hstack()在行上合并(即行数不变，列增加) vstack()在列上合并（列不变，行数增加）
y_data=np.hstack((y_data,y_data1))
print(y_data)

# 2.模型构建，就是一个函数
x=tf.placeholder(dtype=tf.float32,shape=[None,2],name='x')
y=tf.placeholder(dtype=tf.float32,shape=[None,2],name='y')

# 定义一个变量w和b
# w当中的第一个2表示：样本的特征属性个数
# w当中的第二个2表示：样本的目标属性所属的类别数目（有多少个类别）
# b当中的第二个2表示：样本的目标属性所属的类别数目（有多少个类别）
w=tf.Variable(initial_value=tf.zeros(shape=[2,2]),name='w')
b=tf.Variable(initial_value=tf.zeros(shape=[2]),name='b')

# 3.得到一个预测值
actv=tf.nn.softmax(tf.matmul(x,w)+b)

# 4.构建一个损失函数，交叉熵损失函数
# tf.reduce_mean：求均值，当不给定任何axis参数的时候表示求解全部所有数据的均值
# tf.reduce_sum：求和，当参数为矩阵的时候，axis为1表示对每一行求和 与numpy中axis参数的含义一样
# y表示：传入进来的真实值
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv),axis=1))

# 5.以随机梯度下降的方式优化损失函数
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

#输出最终的正确值（实际值与预测值是否一致）
pred=tf.equal(tf.argmax(actv,axis=1),tf.argmax(y,axis=1))
# 正确率（True转化为1，False转换为0）
acc=tf.reduce_mean(tf.cast(pred,tf.float32))

# 总共迭代次数
training_epochs=60
# 批次数量
num_batch=int(N/10)
# 训练迭代次数每隔5词就打印一次信息
display_step=5

# 6.执行阶段
with  tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        #每一轮的损失值
        avg_cost=0
        #打乱数据的顺序，每一次训练都是针对所有的样本
        index=np.random.permutation(N)

        #针对每个batch, 对所有样本分批次
        for i in range(num_batch):
            xy_index=index[i*10:(i+1)*10]
            #构造传入的feeds参数
            feeds={x:x_data[xy_index],y:y_data[xy_index]}
            #进行模型训练
            sess.run(train,feed_dict=feeds)
            #获取每个批次的损失值（均值）
            avg_cost+=sess.run(cost,feeds)

        #训练迭代次数每隔5次 打印信息
        if epoch %display_step==0:
            feed_train={x:x_data,y:y_data}
            train_acc=sess.run(acc,feed_dict=feed_train)
            print('训练迭代次数：%03d/%03d 损失值：%0.9f 训练集上的准确率：%0.3f' %
                  (epoch, training_epochs, avg_cost, train_acc))
    print('模型训练完成')


