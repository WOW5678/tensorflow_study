# -*- coding:utf-8 -*-
'''
如何保存以及恢复神经网络中的变量
'''
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import os
import math
import prettytensor as pt
print(tf.__version__)
print(pt.__version__)

'''
载入数据
'''
from tensorflow.examples.tutorials.mnist import input_data
data=input_data.read_data_sets('data/MNIST/',one_hot=True)
print('SIZE OF:')
print('-Training-Set:\t\t{}'.format(len(data.train.labels)))
print('-Test-Set:\t\t{}'.format(len(data.test.labels)))
print('-Valid-Set"\t\t{}'.format(len(data.validation.labels)))
data.test.cls=np.argmax(data.test.labels,axis=1)
data.validation.cls=np.argmax(data.validation.labels,axis=1)
'''
数据维度
'''
img_size=28
img_size_flat=img_size*img_size
img_shape=(img_size,img_size)
num_channels=1
num_classes=10
'''
用来绘制图片的帮助函数
'''
def plot_images(images,cls_true,cls_pred=None):
    assert len(images)==len(cls_true)==9
    #create figure with 3*3 sub-plot
    fig,axes=plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)

    for i,ax in enumerate(axes.flat):
        #plot images
        ax.imshow(images[i].reshape(img_shape),cmap='binary')
        #show true and predicted classes
        if cls_pred is None:
            xlabel='True:{0}'.format(cls_true[i])
        else:
            xlabel='True:{0},Pred:{1}'.format(cls_true[i],cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
'''
绘制几张图 检验是否有问题
'''
images=data.test.images[:9]
cls_true=data.test.cls[:9]
plot_images(images,cls_true=cls_true)
'''
占位符变量
'''
x=tf.placeholder(tf.float32,shape=[None,img_size_flat],name='x')
x_image=tf.reshape(x,[-1,img_size,img_size,num_channels])
y_true=tf.placeholder(tf.float32,shape=[None,num_classes],name='y_true')
y_true_cls=tf.argmax(y_true,dimension=1)
'''
神经网络
'''
x_pretty=pt.wrap(x_image)
with pt.defaults_scope(activation_fn=tf.nn.relu):
                y_pred,loss=x_pretty.\
                    conv2d(kernel=5,depth=16,name='layer_conv1').\
                    max_pool(kernel=2,stride=2).\
                    conv2d(kernel=5,depth=36,name='layer_conv2').\
                    max_pool(kernel=2,stride=2).\
                    flatten().\
                    fully_connected(size=128,name='layer_fc1').\
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
优化
'''
optimizer=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
y_pred_cls=tf.argmax(y_pred,dimension=1)
#bool型
correct_prediction=tf.equal(y_pred_cls,y_true_cls)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
'''
创建saver-object的对象，它用来保存以及恢复tensorflow图中的所有变量
'''
saver=tf.train.Saver()
#用来保存或者恢复数据的文件夹
save_dir='checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
#用来保存checkpoint文件的路径
save_path=os.path.join(save_dir,'best_validation')
'''
运行tensorflow
'''
session=tf.Session()
def init_variable():
    session.run(tf.global_variables_initializer())
#运行函数来初始化变量
init_variable()
#用来优化迭代的帮助函数
train_batch_size=64
#目前为止在验证集上最佳的准确率
best_validation_accuracy=0.0
#准确率有提升的iteration
last_improvement=0

require_imporement=1000

total_iterations=0
def optimize(num_iterations):
    global total_iterations
    global best_validation_accuracy
    global last_improvement

    start_time=time.time()
    for i in range(num_iterations):
        total_iterations+=1
        x_batch,y_true_batch=data.train.next_batch(train_batch_size)
        feed_dict_train={x:x_batch,y_true:y_true_batch}
        session.run(optimizer,feed_dict=feed_dict_train)
        if (total_iterations%100==0)or (i==(num_iterations-1)):
            acc_train=session.run(accuracy,feed_dict=feed_dict_train)
            acc_validataion,_=validation_accuracy()
            #这种策略称为Early-Stopping。它用来避免神经网络的过拟合。
            if acc_validataion>best_validation_accuracy:
                best_validation_accuracy=acc_validataion
                last_improvement=total_iterations

                #save all variables of tensorflow graph to file
                saver.save(sess=session,save_path=save_path)
                improved_str='*'
            else:
                improved_str=''
            msg='Iter:{0:>6},Train-batch Accruacy:{1:>6.1%},Validation ACC:{2:>6.1%}{3}'
            print(msg.format(i+1,acc_train,acc_validataion,improved_str))
        #如果最近的1000词迭代中 没有准确率的提升
        if total_iterations-last_improvement>require_imporement:
            print('No improvement found in a while,stopping optimization.')
            break
    end_time=time.time()
    time_dif=end_time-start_time
    print('Time usage:'+str(timedelta(seconds=int(round(time_dif)))))
'''
计算验证集上的准确率
'''
def validation_accuracy():
    correct,_=predict_cls_validation()
    return cls_accuracy(correct)
'''
计算样征集上的预测类别
'''
def predict_cls_validation():
    return predict_cls(images=data.validation.images,labels=data.validation.labels,
                       cls_true=data.validation.cls)
'''
计算图像的预测类别
'''
batch_size=256
def predict_cls(images,labels,cls_true):
    num_images=len(images)
    cls_pred=np.zeros(shape=num_images,dtype=np.int)
    i=0
    while i<num_images:
        j=min(i+batch_size,num_images)
        feed_dict={x:images[i:j,:],y_true:labels[i:j,:]}
        cls_pred[i:j]=session.run(y_pred_cls,feed_dict=feed_dict)
        i=j
    correct=(cls_true==cls_pred)
    return correct,cls_pred
'''
计算分类准确率
'''
def cls_accuracy(correct):
    correct_sum=correct.sum()
    acc=float(correct_sum)/len(correct)
    return acc,correct_sum
'''
用来绘制测试集上错误样本的帮助函数
'''
def plot_example_errors(cls_pred,correct):
    incorrect=(correct==False)
    images=data.test.images[incorrect]
    cls_pred=cls_pred[incorrect]
    cls_true=data.test.cls[incorrect]
    plot_images(images=images[:9],cls_true=cls_true[:9],cls_pred=cls_pred[:9])
'''
绘制混淆矩阵
'''
def plot_confusion_matrix(cls_pred):
    cls_true=data.test.cls
    cm=confusion_matrix(y_true=cls_true,y_pred=cls_pred)
    print(cm)
    plt.imshow(cm)
    plt.colorbar()
    tick_marks=np.arange(num_classes)
    plt.xticks(tick_marks,range(num_classes))
    plt.yticks(tick_marks,range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

'''
打印测试集上的分类准确率
'''
def print_test_accuracy(show_example_errors=False,show_confusion_matrix=False):
    correct,cls_pred=predict_cls(data.test.images,data.test.labels,data.test.cls)
    num_images=len(correct)
    acc,num_correct=cls_accuracy(correct)
    msg='Accuracy on Test-Set:{0:.1%}={1}/{2}'
    print(msg.format(acc,num_correct,num_images))
    if show_example_errors:
        print('Examples errors:')
        plot_example_errors(cls_pred=cls_pred,correct=correct)
    if show_confusion_matrix:
        print('Confusion Matrix:')
        plot_confusion_matrix(cls_pred=cls_pred)
'''
绘制卷积权重的帮助函数
'''
def plot_conv_weights(weights,input_channel=0):
    w=session.run(weights)
    print('Mean:{0:.5f},Stdev:{1:.5f}'.format(w.mean(),w.std()))
    w_min=np.min(w)
    w_max=np.max(w)
    num_filters=w.shape[3]
    num_grids=math.ceil(math.sqrt(num_filters))
    fig,axes=plt.subplots(num_grids,num_grids)
    for i,ax in enumerate(axes.flat):
        if i<num_filters:
            img=w[:,:,input_channel,i]
            ax.imshow(img,vmin=w_min,vmax=w_max,interpolation='nearest',cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

'''
优化之前的性能
'''
print_test_accuracy()
plot_conv_weights(weights=weights_conv1)
'''
1000词典迭代优化后的性能
'''
optimize(num_iterations=1000)
print_test_accuracy(show_example_errors=True,show_confusion_matrix=True)
plot_conv_weights(weights=weights_conv1)
''''
再次初始化变量
'''
print('==================================')
init_variable()
print_test_accuracy()
plot_conv_weights(weights=weights_conv1)
'''
恢复最好的变量
'''
saver.restore(sess=session,save_path=save_path)
print_test_accuracy()
'''
关闭session回话
'''
session.close()