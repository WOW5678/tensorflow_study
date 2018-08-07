# -*- coding:utf-8 -*-
'''
function:
BILSTM的tensorflow实现，使用的数据集是minist数据集
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('data/',one_hot=True)

learing_rate=0.1
max_axamples=4000
display_size=10
batch_size=128

# 每一行作为一个输入，输入到网络中
n_inputs=28
n_steps=28
n_hidden=32
n_classes=10

x=tf.placeholder(tf.float32,shape=[None,n_steps,n_inputs])
y=tf.placeholder(tf.float32,shape=[None,n_classes])

#全连接层的参数，调用basicLSTMCell这个op,参数已经包含在内部了，不需要再定义
weight=tf.Variable(tf.random_normal([2*n_hidden,n_classes]))
bias=tf.Variable(tf.random_normal([n_classes]))

def BIRNN(x,weights,biases):
    #[1,0,2]
    x=tf.transpose(x,[1,0,2])
    x=tf.reshape(x,[-1,n_inputs]) # x.shape=[batch_size*n_steps,n_inputs]
    x=tf.split(x,n_steps)

    # 调用现成的BasicLSTMCell，建立两条完全一样，又独立的LSTM结构
    lstm_qx=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    lstm_hx=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
    # 两个完全一样的LSTM结构输入到static_birectional_rnn中，由这个op来管理双向计算过程
    ouputs,_,__=tf.contrib.rnn.static_bidirectional_rnn(lstm_qx,lstm_hx,x,dtype=tf.float32)
    # 最后来一个全连接层分类预测
    return tf.matmul(ouputs[-1],weights)+biases

pred=BIRNN(x,weight,bias)
#计算损失、优化、精度（老套路）
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learing_rate).minimize(cost)
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()

# RNN图
with tf.Session() as sess:
    sess.run(init)
    step=1
    while step*batch_size<max_axamples:
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        batch_x=batch_x.reshape(batch_size,n_steps,n_inputs)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step%display_size==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            loss=sess.run(cost, feed_dict = {x:batch_x, y:batch_y})
            print ('Iter' + str(step * batch_size) + ', Minibatch Loss= %.6f' % (loss) + ', Train Accurancy= %.5f' % acc)
        step += 1
    print ("Optimizer Finished!")

    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape(-1, n_steps, n_inputs)
    test_label = mnist.test.labels[:test_len]
    print ('Testing Accurancy:%.5f' % (sess.run(accuracy, feed_dict={x: test_data, y: test_label})))

    Coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=Coord)





