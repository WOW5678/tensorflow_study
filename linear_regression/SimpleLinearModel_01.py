# -*- coding:utf-8 -*-
'''
在tensorflow中使用一个简单的线性模型的工作流程
使用MINIST的手写数字图片数据集
使用tensorflow定义并优化了一个数学模型
并画出最终的结果
'''

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

print(tf.__version__)

'''
载入数据
'''
from tensorflow.examples.tutorials.mnist import input_data
data=input_data.read_data_sets('data/MNIST/',one_hot=True)
print('size of:')
print('-Training-set:\t\t{}'.format(len(data.train.labels)))
print('-Test-set:\t\t{}'.format(len(data.test.labels)))
print('-Validation-set:\t\t{}'.format(len(data.validation.labels)))
print(data.test.labels[0:5,:])
#通过获取最大元素的索引将one-hot编码的向量转换成一个单独的数字
data.test.cls=np.array([label.argmax() for label in data.test.labels])
print(data.test.cls[0:5])

'''
数据维度
'''
img_size=28
img_size_flat=img_size*img_size
img_shape=(img_size,img_size)
num_classes=10

'''
用来绘制图像的帮助函数
'''
def plot_images(images,cls_true,cls_pred=None):
    #确保传入的是9张图片
    assert len(images)==len(cls_true)==9
    #创建图片 with 3*3个子图
    fig,axes=plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)

    for i,ax in enumerate(axes.flat):
        #plot image
        ax.imshow(images[i].reshape(img_shape),cmap='binary')

        #show true and predicted classes
        if cls_pred is None:
            xlabel='True:{0}'.format(cls_true[i])
        else:
            xlabel='True:{0},Pred:{1}'.format(cls_true[i],cls_pred[i])

        ax.set_xlabel(xlabel)
        #移除图片中的刻度
        ax.set_xticks([])
        ax.set_yticks([])
    #显示图像
    plt.show()

'''
绘制几张图片验证以上函数
'''
images=data.test.images[0:9]
cls_true=data.test.cls[0:9]
plot_images(images,cls_true=cls_true)

'''
tensorflow占位符
'''
x=tf.placeholder(tf.float32,[None,img_size_flat])
#真实的类别 one-hot编码
y_true=tf.placeholder(tf.float32,[None,num_classes])
#真实的类别 class number  注意这个的data type为int64 否则会报错
y_true_cls=tf.placeholder(tf.int64,[None])
weights=tf.Variable(tf.zeros([img_size_flat,num_classes]))
#行向量 1*num_classes
biases=tf.Variable(tf.zeros([num_classes]))
logist=tf.matmul(x,weights)+biases
#使用softmax函数将数值限制在0-1之间
y_pred=tf.nn.softmax(logist)
#从y_pred矩阵中选取每行最大元素的索引值就是预测的类别
y_pred_cls=tf.argmax(y_pred,dimension=1)

'''
优化损失函数
'''
#计算交叉熵，注意它使用的是logits的值，因为在它内部也计算了softmax
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logist,labels=y_true)
#利用所有图像分类交叉熵的均值 作为损失值
cost=tf.reduce_mean(cross_entropy)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

'''
性能度量
'''

#这是一个布尔值，代表预测类型是否等于每张图片的真实类型
correct_prediction=tf.equal(y_pred_cls,y_true_cls)
#先将布尔值转换成浮点型向量 flase->0,true->1,计算这些值的平均值以此计算准确度
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

''''
运行tensorflow
'''
session=tf.Session()
#初始化变量
session.run(tf.global_variables_initializer())

batch_size=100
#优化迭代来逐步提升模型的权重和偏见
def optimize(num_iterations):
    for i in range(num_iterations):
        x_batch,y_true_batch=data.train.next_batch(batch_size)
        feed_dict_train={x:x_batch,y_true:y_true_batch}
        session.run(optimizer,feed_dict=feed_dict_train)

'''
展示性能的帮助函数
'''
feed_dict_test={x:data.test.images,y_true:data.test.labels,y_true_cls:data.test.cls}
#用来打印测试集分类准确度的函数
def print_accuracy():
    acc=session.run(accuracy,feed_dict=feed_dict_test)
    print('The accuracy on test-data:{0:.1%}'.format(acc))

#用sklearn打印混淆矩阵
def print_confusion_matrix():
    cls_true=data.test.cls
    cls_pred=session.run(y_pred_cls,feed_dict=feed_dict_test)
    #得到混淆矩阵
    cm=confusion_matrix(y_true=cls_true,y_pred=cls_pred)
    print(cm)
    #将混淆矩阵绘制成图像
    plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)

    #调整图像
    plt.tight_layout()
    plt.colorbar()
    tick_marks=np.array(num_classes)
    plt.xticks(tick_marks,range(num_classes))
    plt.yticks(tick_marks,range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

'''
绘制测试集中误分类图像的函数
'''
def plot_example_error():
    correct,cls_pred=session.run([correct_prediction,y_pred_cls],feed_dict=feed_dict_test)
    incorrect=(correct==False)
    #从测试集中获取这些误分类的图像
    images=data.test.images[incorrect]
    #获取这些图片被误分成的类别
    cls_pred=cls_pred[incorrect]
    #真实的类别
    cls_true=data.test.cls[incorrect]
    #绘制前9张图片
    plot_images(images=images[:9],cls_true=cls_true[:9],cls_pred=cls_pred[:9])

'''
绘制模型权重的帮助函数
'''
def plot_weights():
    w=session.run(weights)
    w_min=w.min()
    w_max=w.max()

    fig,axes=plt.subplots(3,4)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)

    for i,ax in enumerate(axes.flat):
        #仅仅使用前10张图的权重
        if i <10:
            #w.shape==(img_size,10)
            image=w[:,i].reshape(img_shape)
            ax.set_label('weights:{0}'.format(i))
            ax.imshow(image,vmin=w_min,vmax=w_max,cmap='seismic')

    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()

'''
优化之前的性能
'''
print_accuracy()
plot_example_error()

'''
1次迭代优化后的性能
'''
optimize(num_iterations=1)
print_accuracy()
plot_weights()

'''
10次迭代优化后的性能
'''
optimize(num_iterations=10)
print_accuracy()
plot_weights()

''''
1000次迭代优化后的性能
'''
optimize(num_iterations=1000)
print_accuracy()
plot_weights()




