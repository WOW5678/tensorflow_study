# -*- coding:utf-8 -*-
'''
learning pretty tensor
'''
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
from sklearn.metrics import confusion_matrix
import prettytensor as pt
print(tf.__version__)
'''
载入数据
'''
from tensorflow.examples.tutorials.mnist import input_data
data=input_data.read_data_sets('data/MNIST/',one_hot=True)
print('size of :')
print('-Traing-set:\t\t{}'.format(len(data.train.labels)))
print('-Test-set:\t\t{}'.format(len(data.test.labels)))
print('-Validation-set:\t\t{}'.format(len(data.validation.labels)))
data.test.cls=np.argmax(data.test.labels,axis=1)
'''
数据维度
'''
img_size=28
img_size_flat=img_size*img_size
img_shape=(img_size,img_size)
num_channels=1
num_classes=10
'''
用来绘制图像的帮助函数
'''
def plot_image(images,cls_true,cls_pred=None):
    assert len(images)==len(cls_true)==9
    fig,axes=plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    for i,ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape),cmap='binary')
        if cls_pred is None:
            xlabel='True:{0}'.format(cls_true[i])
        else:
            xlabel='True:{0},Pred:{1}'.format(cls_true[i],cls_pred[i])

        ax.set_xlabel(xlabel=xlabel)
    #remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
'''
绘制几张图检查函数是否正确
'''
images=data.test.images[:9]
cls_true=data.test.cls[:9]
plot_image(images=images,cls_true=cls_true)
'''
占位符变量
'''
x=tf.placeholder(tf.float32,shape=[None,img_size_flat],name='x')
x_image=tf.reshape(x,[-1,img_size,img_size,num_channels])
y_true=tf.placeholder(tf.float32,shape=[None,num_classes],name='y_true')
y_true_cls=tf.argmax(y_true,dimension=1)
'''
tensorflow实现
'''
def new_weight(shape):
    return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05,shape=[length]))
def new_conv_layer(input,#上一层
                   num_input_channals,#num_channels in pre. layer
                   filter_size,#weidth and height of filter
                   num_filters,#num of filter
                   use_polling=True):#use 2*2 max-pooling
    shape=[filter_size,filter_size,num_input_channals,num_filters]
    weights=new_weight(shape=shape)
    biases=new_biases(length=num_filters)
    layer=tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')
    layer+=biases
    if use_polling:
        layer=tf.nn.max_pool(value=layer,
                             ksize=[1,2,2,1],
                             strides=[1,2,2,1],
                             padding='SAME')

    layer=tf.nn.relu(layer)
    return layer,weights
def flatten_layser(layer):
    layer_shape=layer.get_shape()
    num_features=layer_shape[1:4].num_elements()
    layer_flat=tf.reshape(layer,[-1,num_features])
    return layer_flat,num_features
'''
全连接层
'''
def new_fc_layer(input,
                 num_inputs,
                 num_outputs,
                 use_rele=True):
    weights=new_weight(shape=[num_inputs,num_outputs])
    biases=new_biases(length=num_outputs)
    layer=tf.matmul(input,weights)+biases
    if use_rele:
        layer=tf.nn.relu(layer)
    return layer
'''
prettyTensor实现
'''
x_pretty=pt.wrap(x_image)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred,loss=x_pretty.\
        conv2d(kernel=5,depth=16,name='layer_conv1').\
        max_pool(kernel=2,stride=2).\
        conv2d(kernel=5,depth=36,name='layer_conv2').\
        max_pool(kernel=2,stride=2).\
        flatten().\
        fully_connected(size=128,name='kayer_fc1').\
        softmax_classifier(num_classes=num_classes,labels=y_true)
'''
获取权重
'''
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name,reuse=True):
        variable=tf.get_variable('weights')
    return variable
weights_conv1=get_weights_variable(layer_name='layer_conv1')
weights_conv2=get_weights_variable(layer_name='layer_conv2')
'''
优化方法
'''
optimizer=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
y_pred_cls=tf.argmax(y_pred,dimension=1)
#布尔值
correct_pred=tf.equal(y_pred_cls,y_true_cls)
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
''''
运行tensorflow
'''
session=tf.Session()
session.run(tf.global_variables_initializer())
train_batch_size=64
total_iterations=0
def optimize(num_iterations):
    global  total_iterations
    start_time=time.time()
    for i in range(total_iterations,total_iterations+num_iterations):
        x_batch,y_true_batch=data.train.next_batch(train_batch_size)
        feed_dict_train={x:x_batch,y_true:y_true_batch}
        session.run(optimizer,feed_dict=feed_dict_train)
        if i %100==0:
            acc=session.run(accuracy,feed_dict=feed_dict_train)
            msg='Optimizaton Iteration:{0:>6},Training Accuracy:{1:>6.1%}'
            print(msg.format(i+1,acc))
    total_iterations+=num_iterations
    end_time=time.time()
    time_dif=end_time-start_time
    print('Time usage:'+str(timedelta(seconds=int(round(time_dif)))))
'''
用来绘制错误样本的帮助函数
'''
def plot_example_errors(cls_pred,correct):
    incorrect=(correct==False)
    images=data.test.images[incorrect]
    cls_pred=cls_pred[incorrect]
    cls_true=data.test.cls[incorrect]
    plot_image(images=images[:9],
               cls_true=cls_true[:9],
               cls_pred=cls_pred[:9])
'''
绘制混淆矩阵的帮助函数
'''
def plot_confusion_matrix(cls_pred):
    cls_true=data.test.cls
    cm=confusion_matrix(y_true=cls_true,y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks=np.arange(num_classes)
    plt.xticks(tick_marks,range(num_classes))
    plt.yticks(tick_marks,range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
'''
打印测试集上的分类准确du 
'''
test_batch_size=256
def print_test_accuracy(show_example_errors=False,show_confusion_matrix=False):
    num_test=len(data.test.labels)
    cls_pred=np.zeros(shape=num_test,dtype=np.int)
    i=0
    while i<num_test:
        j=min(i+test_batch_size,num_test)
        images=data.test.images[i:j,:]
        labels=data.test.labels[i:j,:]
        feed_dict_test={x:images,y_true:labels}
        cls_pred[i:j]=session.run(y_pred_cls,feed_dict=feed_dict_test)
        i=j
    cls_true=data.test.cls
    correct=(cls_true==cls_pred)
    correct_sum=correct.sum()
    acc=float(correct_sum)/num_test
    msg='Accuracy on Test-Set:{0:.1%}={1}/{2}'
    print(msg.format(acc,correct_sum,num_test))

    if show_example_errors:
        print('Examples errors:')
        plot_example_errors(cls_pred=cls_pred,correct=correct)
    if show_confusion_matrix:
        print('Confusion Matrix:')
        plot_confusion_matrix(cls_pred=cls_pred)
optimize(1000)
print_test_accuracy(show_example_errors=True,show_confusion_matrix=True)



