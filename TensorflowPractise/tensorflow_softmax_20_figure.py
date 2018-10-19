# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/19 0019 上午 9:36
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 使用tensorflow实现softmax分类问题，并画图
"""
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


# TensorFlow实现Softmax分类问题代码案例

# 1、构造数据
N = 200
np.random.seed(20)
x_data = np.random.normal(loc=0, scale=2, size=(N, 2))
y_data = np.dot(x_data, [[5], [-3]])
# print(y_data)
# 二值化
y_data[y_data > 0] = 1
y_data[y_data <= 0] = 0
# print(y_data)

y_data1 = 1 - y_data
# hstack()在行上合并  vstack()在列上合并
y_data = np.hstack((y_data, y_data1))
print(y_data.shape)
print(y_data)

#构建最终画图需要的数据
t1=np.linspace(-8,10,100,dtype=np.float32)
t2=np.linspace(-8,10,100,dtype=np.float32)
xv,yv=np.meshgrid(t1,t2)
# dstack()：堆栈数组按顺序深入（沿第三维）
x_test=np.dstack((xv.flat,yv.flat))[0]
print(x_test.shape)
# 2、模型构建 就是一个函数
# 传入部分数据进行模型参数更新 一般使用MBGD 设置占位符
# None的意思是行维度未知（也就是可以传入任意的样本条数）
# x当中的2表示：样本的特征属性是2个特征
# y当中的2表示：所属类别的数目，处理的是几分类问题这里就是几
x = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='x')
y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='y')

# 定义一个变量w和变量b
# w当中的第一个2表示：样本的特征属性个数
# w当中的第二个2表示：样本的目标属性所属的类别数目（有多少个类别）
# b当中的第二个2表示：样本的目标属性所属的类别数目（有多少个类别）
# tf.set_random_seed(10)
w = tf.Variable(initial_value=tf.zeros(shape=[2, 2]), name='w')
b = tf.Variable(initial_value=tf.zeros(shape=[2]), name='b')


# 3、得到一个预测值
# act是通过softmax函数转换后的一个概率值（矩阵形式）
actv = tf.nn.softmax(tf.matmul(x, w) + b)

# 4、构建一个损失函数 交叉熵损失函数
# tf.reduce_mean：求均值，当不给定任何axis参数的时候表示求解全部所有数据的均值
# tf.reduce_sum：求和，当参数为矩阵的时候，axis为1表示对每一行求和 与numpy中axis参数的含义一样
# y表示：传入进来的真实值
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), axis=1))

# 5、以随机梯度下降的方式优化损失函数 在训练优化的过程中让哪个函数最小
# 学习率：过大可能不会收敛，跑出去了；过小收敛速度太慢
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss=cost)

# 输出最终的正确率（实际值与预测值是否一致）
# y为实际值 actv为预测值
# tf.argmax：对矩阵按行或按列计算最大值对应的下标
pred = tf.equal(tf.argmax(actv, axis=1), tf.argmax(y, axis=1))
# 正确率 (True转换为1 False转换为0)
acc = tf.reduce_mean(tf.cast(pred, dtype=tf.float32))


# 共总训练迭代次数
training_epochs = 60
# 批次数量
num_batch = int(N/10)
# 训练迭代次数每隔5次就打印信息
display_step = 5

# 6、执行阶段
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        # 开始训练
        avg_cost = 0
        # 打乱数据的顺序 每一次训练都是针对所有的样本
        index = np.random.permutation(N)
        # 下面的for循环 就是对所有样本进行分批次
        for i in range(num_batch):
            # 获取传入进行模型训练的数据对应索引
            xy_index = index[i * 10: (i + 1) * 10]
            # 构造传入的feeds参数
            feeds = {x: x_data[xy_index], y: y_data[xy_index]}
            # 进行模型训练
            sess.run(train, feed_dict=feeds)
            # 获取每批次损失函数值（均值）
            avg_cost += sess.run(cost, feed_dict=feeds)

        # 训练迭代次数每隔5次 打印信息
        if epoch % display_step == 0:
            feeds_train = {x: x_data, y: y_data}
            train_acc = sess.run(acc, feed_dict=feeds_train)
            print('训练迭代次数：%03d/%03d 损失值：%0.9f 训练集上的准确率：%0.3f' % (epoch, training_epochs, avg_cost, train_acc))

    print('训练过程完成！！！')
    # 对于画图的测试数据进行预测
    y_test=sess.run(actv ,feed_dict={x:x_test})
    #按行比较哪个大，返回其索引
    y_test=np.argmax(y_test,axis=1)
    y_hat=y_test.reshape(xv.shape)

# 7.画图
plt.figure(figsize=(12,6),facecolor='w')
plt.xlabel(u'X1',fontsize=16)
plt.ylabel(u'X2',fontsize=16)
cm_light=mpl.colors.ListedColormap(['#bde1f5', '#f7cfc6'])
# 测试集上的预测值 画图形成区域
plt.pcolormesh(xv, yv, y_hat, cmap=cm_light)
# 训练集上的数据 x1 x2
plt.scatter(x_data[y_data[:, 0] == 0][:, 0], x_data[y_data[:, 0] == 0][:, 1], s=50, marker='+', c='red')
plt.scatter(x_data[y_data[:, 0] == 1][:, 0], x_data[y_data[:, 1] == 0][:, 1], s=50, marker='o', c='blue')

#有网格
plt.grid(True)
plt.title(u'TensorFlow实现Softmax分类问题', fontsize=20)
plt.show()