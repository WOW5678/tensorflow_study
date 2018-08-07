# -*- coding:utf-8 -*-
'''
加载训练好的模型
在测试阶段 将前两层的结果单独保存下来，再将中间结果传入到模型中进行后面几层的训练，最终得到预测结果
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist  import input_data
import pickle

# 加载数据
mnist=input_data.read_data_sets('data\mnist',one_hot=True)

def set_weights(x,shape,activation_fun=None):
    w=tf.Variable(tf.zeros(shape))
    b=tf.Variable(tf.zeros((1,shape[1])))
    w_plus_b=tf.matmul(x,w)+b
    if activation_fun:
        output=activation_fun(w_plus_b)
    else:
        output=w_plus_b
    return w,b,output

# define the classification model
x=tf.placeholder(tf.float32,[None,784])
#真实值
y=tf.placeholder(tf.float32,[None,10])
w1,b1,h1=set_weights(x,[784,50],activation_fun=tf.nn.relu)
w2,b2,h2=set_weights(h1,[50,20],activation_fun=tf.nn.relu)
w3,b3,y_=set_weights(h2,[20,10],activation_fun=None)

correct_prediction=tf.equal(tf.argmax(y_,1),tf.argmax(y,1)) #计算预测值与真实值是否相等
#布尔值 转换成浮点数，并取平均值 得到的就是准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,save_path='checkpoints/final_model')
    acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
    print('accuracy:{}'.format(acc))
    h2_vlaue=sess.run(h2,feed_dict={x:mnist.test.images,y:mnist.test.labels})
    w3_value,b3_value=sess.run([w3,b3],feed_dict={x:mnist.test.images,y:mnist.test.labels})
    print('h2_value:{}'.format(h2_vlaue.shape))
    with open('h2.pkl','wb') as f:
        pickle.dump(h2_vlaue,f)
    with open('w3-b3.pkl','wb') as f:
        pickle.dump([w3_value,b3_value],f)
    y_predict=sess.run(y_,feed_dict={x:mnist.test.images,y:mnist.test.labels})
    print('y_predict:',y_predict)


