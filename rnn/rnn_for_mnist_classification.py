# -*- coding:utf-8 -*-
#功能：实现一个最简单的RNN模型
from tensorflow.examples.tutorials.mnist import  input_data
import  tensorflow as tf
import numpy as np

#加载数据集
mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
print (u"输入数据：")
print (mnist.train.images)
print (u'输入数据的格式：')
# 每副图像的格式为28×28
print (mnist.train.images.shape)

#将其装换成图像 观察如下图所示
import pylab

im=mnist.train.images[1]
im=im.reshape((-1,28))
pylab.imshow(im)
pylab.show()

#如果我们要用RNN来训练这个网络的话，则应该选择 n-input=28 n_steps=28的结构
# a=np.asarray(range(20))
# b=a.reshape((-1,2,2))
# print u'生成一列数据'
# print a
# print u'reshape函数的效果'
# print b
#
# c=np.transpose(b,[1,0,2])
# d=c.reshape(-1,2)
# print '-------c--------'
# print c
# print '-------d--------'
# print d

#定义一些模型的参数
learning_rate=0.001
training_iters=200
batch_size=128
display_step=100

#网络的参数
n_input=28
n_step=28
n_hidden=120
n_classes=10

#构建网络
x=tf.placeholder('float32',[None,n_step,n_input])
y=tf.placeholder('float32',[None,n_classes])
#定义权重
weights={
    'hidden':tf.Variable(tf.random_normal([n_input,n_hidden])),
    'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))
}
#定义偏向
biases={
    'hidden':tf.Variable(tf.random_normal([n_hidden])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

#首先创建一个ceil 这里需要一个参数是隐藏单元的个数 n_hidden ,在创建完成后对其进行初始化
lstm_ceil=tf.nn.rnn_cell.BasicLSTMCell(n_hidden,forget_bias=0.0,state_is_tuple=True)
_state=lstm_ceil.zero_state(batch_size,tf.float32)
#为了使得原始数据的输入和模型匹配，我们对数据进行一系列变换，变换结果如下

a1=tf.transpose(x,[1,0,2])
a2=tf.reshape(a1,[-1,n_input])
a3=tf.matmul(a2,weights['hidden']+biases['hidden'])
a4=tf.split(a3,n_step,0)

print('a4:',a4)
outputs,state=tf.nn.static_rnn(lstm_ceil,a4,initial_state=_state)
print (outputs[-1])
print('state:',state)

a5=tf.matmul(outputs[-1],weights['out'])+biases['out']
print ('a5:')
print (a5)

#定义损失函数，使用梯度下降求最优
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=a5,labels=y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred=tf.equal(tf.argmax(a5,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init=tf.initialize_all_variables()

#进行模型的训练
sess=tf.Session()
sess.run(init)
step=1
while step *batch_size<training_iters:
    batch_xs,batch_ys=mnist.train.next_batch(batch_size)
    batch_xs=batch_xs.reshape((batch_size,n_step,n_input))
    sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys})
    if step %display_step==0:
        acc=sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys})
        loss=sess.run( cost,feed_dict={x:batch_xs,y:batch_ys})
        print ('iter'+str(step*batch_size)+',minibatch Loss='+':{:.6f}'.format(loss)+', Training accuracy='+'{:.5f}'.format(acc))
        #state是一个二元的元祖，第一维保存状态ci,第二维保存hi
        #因为n_hidden=120 所以为（128,120）
        state_01=sess.run(state,feed_dict={x:batch_xs,y:batch_ys})

    step+=1

print ('optimization Finished!')

#测试模型的准确率
test_len=batch_size
#仅使用前128个数据进行测试
test_data=mnist.test.images[:test_len].reshape((-1,n_step,n_input))
test_label=mnist.test.labels[:test_len]

#评估模型
correct_pred=tf.equal(tf.argmax(a5,1),tf.argmax(y,1))
print ('testing accuracy:',sess.run(accuracy,feed_dict={x:test_data,y:test_label}))
