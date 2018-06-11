# -*- coding:utf-8 -*-
''''
使用tensorflow实现一个简单的卷积神经网络
'''

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

print(tf.__version__)

'''
神经网络的配置
'''
#卷基层1
filter_size1=5
num_filters1=16

#卷基层2
filter_size2=5
num_filters2=36

#全连接层
fc_size=128

'''
载入数据
'''
from tensorflow.examples.tutorials.mnist import input_data
data=input_data.read_data_sets('data/MNIST/',one_hot=True)
print('-Training-set:\t\t{}'.format(len(data.train.labels)))
print('-Test-set:\t\t{}'.format(len(data.test.labels)))
print('-Validation-set:\t\t{}'.format(len(data.validation.labels)))
#每行的最大值的索引就是最终的类别number
data.test.cls=np.argmax(data.test.labels,axis=1)

''''
数据维度
'''
img_size=28
#图像存到一维数组中的长度
img_size_flat=img_size*img_size
img_shape=(img_size,img_size)

num_channels=1
num_classes=10

'''
用来绘制图片的帮助函数
'''
def plot_images(images,cls_true,cls_pred=None):
    assert len(images)==len(cls_true)==9

    #创建3*3规格的图片
    fig,axes=plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)

    for i,ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape),cmap='binary')

        #展示真实的和预测的类别
        if cls_pred is None:
            xlabel='True:{0}'.format(cls_true[i])
        else:
            xlabel='True:{0}, Pred:{1}'.format(cls_true[i],cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

'''
验证绘制图片的函数是否有问题
'''
images=data.test.images[:9]
cls_true=data.test.cls[:9]
plot_images(images,cls_true)

'''
创建新变量的帮助函数
'''
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05,shape=[length]))

'''
创建卷积层的帮助函数
'''
def new_conv_layer(input,num_input_channels,filter_size,num_filters,use_pooling=True):
    shape=[filter_size,filter_size,num_input_channels,num_filters]
    weights=new_weights(shape=shape)
    biases=new_biases(length=num_filters)
    #padding='SAME' means the input image is padded with zeros so the size of the output is the same
    layer=tf.nn.conv2d(input=input,filter=weights,strides=[1,1,1,1],padding='SAME')

    #add the biases to the results of convolution. A bias-value is added to each filter-channel
    layer+=biases

    if use_pooling:
        layer=tf.nn.max_pool(value=layer,
                             ksize=[1,2,2,1],
                             strides=[1,2,2,1],
                             padding='SAME')
    layer=tf.nn.relu(layer)
    return layer,weights
'''
转换一个层的帮助函数
'''
def flatten_layer(layer):
    #get the shape of input layer
    layer_shape=layer.get_shape()
    #the shape of input layer is assumed to be:
    #layer_shape=[num_images,img_height,img_width,num_channels]
    #the number of features is:img_height*img_width*num_channels
    num_features=layer_shape[1:4].num_elements()
    layer_flat=tf.reshape(layer,[-1,num_features])

    #the shape of the flattened layer is now:
    #[num_images,img_heights*img_weight*num_channels]

    return layer_flat,num_features
''''
创建一个全连接层的帮助函数
'''
def new_fc_layer(input,num_inputs,num_outputs,use_rule=True):
    weights=new_weights(shape=[num_inputs,num_outputs])
    biases=new_biases(length=num_outputs)
    layer=tf.matmul(input,weights)+biases
    if use_rule:
        layer=tf.nn.relu(layer)
    return layer

'''
占位符变量
'''
x=tf.placeholder(tf.float32,shape=[None,img_size_flat],name='x')
x_image=tf.reshape(x,[-1,img_size,img_size,num_channels])
#真实标签
y_true=tf.placeholder(tf.float32,shape=[None,num_classes],name='y_true')
#每行的最大值 为真实标签number
y_true_cls=tf.argmax(y_true,dimension=1)

'''
卷基层1
'''
layer_conv1,weights_conv1=new_conv_layer(input=x_image,num_input_channels=num_channels,filter_size=filter_size1,num_filters=num_filters1,use_pooling=True)
print(layer_conv1)

'''
卷基层2
'''
layer_conv2,weights_conv2=new_conv_layer(input=layer_conv1,num_input_channels=num_filters1,filter_size=filter_size2,num_filters=num_filters2,use_pooling=True)
print(layer_conv2)

'''
转换层
'''

layer_flat,num_features=flatten_layer(layer_conv2)
print(layer_flat)
''''
全连接层1
'''
layer_fc1=new_fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=fc_size,use_rule=True)
print(layer_fc1)

'''
全连接层2
'''
layer_fc2=new_fc_layer(input=layer_fc1,num_inputs=fc_size,num_outputs=num_classes,use_rule=True)
print(layer_fc2)

''''
预测类别
'''
#对输出使用softmax进行归一化
y_pred=tf.nn.softmax(layer_fc2)
#类别数字是最大元素的索引
y_pred_cls=tf.argmax(y_pred,dimension=1)

'''
优化损失函数
'''
#为每副图像计算交叉熵
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
cost=tf.reduce_mean(cross_entropy)
#使用梯度下降的变体AdamOptimizer 执行优化
optimizer=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
'''
性能度量
'''
#返回的布尔值
correct_prediction=tf.equal(y_pred_cls,y_true_cls)
#将布尔值转换成浮点型，计算平均值 来计算分类的准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

'''
运行tensorflow
'''
session=tf.Session()
#初始化变量
session.run(tf.global_variables_initializer())
train_batch_size=64
'''
用来优化迭代的帮助函数
'''
total_iteration=0

def optimize(num_iterations):
    global total_iteration
    start_time=time.time()
    for i in range(total_iteration,total_iteration+num_iterations):
        x_batch,y_true_batch=data.train.next_batch(train_batch_size)
        feed_dict_train={x:x_batch,y_true:y_true_batch}
        session.run(optimizer,feed_dict=feed_dict_train)

        if i%100==0:
            acc=session.run(accuracy,feed_dict=feed_dict_train)
            _y_true_cls,_y_pred_cls=session.run([y_true_cls,y_pred_cls],feed_dict=feed_dict_train)
            msg='Optimization Iteration:{0:>6},Training Accuracy:{1:>6.1%}'
            print(msg.format(i+1,acc))
            print('y_pred_cls:',_y_pred_cls)
            print('y_true_cls:',_y_true_cls)
    total_iteration+=num_iterations
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

    #画图
    plot_images(images=images[:9],cls_true=cls_true[:9],cls_pred=cls_pred[:9])
    plt.show()

'''
绘制混淆矩阵的帮助函数
'''
def plot_confusion_matrix(cls_pred):
    cls_true=data.test.cls
    cm=confusion_matrix(y_true=cls_true,y_pred=cls_pred)
    print(cm)

    #plot
    plt.matshow(cm)
    plt.colorbar()
    tick_marks=np.arange(num_classes)
    plt.xticks(tick_marks,range(num_classes))
    plt.yticks(tick_marks,range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()

'''
打印测试集上的准确率
'''
test_batch_size=128
def print_test_accuracy(show_example_errors=False,show_confusion_matrix=False):
    num_test=len(data.test.images)
    cls_pred=np.zeros(shape=num_test,dtype=np.int)
    i=0

    while i <num_test:
        j=min(i+test_batch_size,num_test)
        images=data.test.images[i:j,:]
        labels=data.test.labels[i:j,:]
        feed_dict={x:images,y_true:labels}

        cls_pred[i:j]=session.run(y_pred_cls,feed_dict=feed_dict)

        i=j

    cls_true=data.test.cls
    correct=(cls_true==cls_pred)
    correct_sum=correct.sum()
    acc=float(correct_sum)/num_test

    #打印准确率
    msg='Accuracy on Test-set:{0:.1%}={1}/{2}'
    print(msg.format(acc,correct_sum,num_test))

    if show_example_errors:
        print('Example errors:')
        plot_example_errors(cls_pred=cls_pred,correct=correct)
    if show_confusion_matrix:
        print('confusion matrix:')
        plot_confusion_matrix(cls_pred=cls_pred)

'''
优化之前的性能
'''
print_test_accuracy()
'''
1次迭代后的性能
'''
optimize(num_iterations=1)
print_test_accuracy()

'''
100次迭代优化后的性能
'''
optimize(num_iterations=99)#我们已经在上面迭代了1次
print_test_accuracy()

'''
1000次迭代优化后的性能
'''
optimize(num_iterations=900)
print_test_accuracy(show_example_errors=True,show_confusion_matrix=True)
