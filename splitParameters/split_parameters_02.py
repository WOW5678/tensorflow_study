# -*- coding:utf-8 -*-
'''
加载训练好的模型
获取中间某层的结果
进行后续的预测运算
'''
import tensorflow  as tf
import pickle
def set_weights(x,activation_fun=None):
    with open('w3-b3.pkl', 'rb') as f:
        w3, b3 = pickle.load(f)
    w_plus_b=tf.matmul(x,w3)+b3
    if activation_fun:
        output=activation_fun(w_plus_b)
    else:
        output=w_plus_b
    return output

# define the classification model
x=tf.placeholder(tf.float32,[None,784])
#真实值
y=tf.placeholder(tf.float32,[None,10])
h2=tf.placeholder(tf.float32,[None,20])
y_=set_weights(h2,activation_fun=None)

correct_prediction=tf.equal(tf.argmax(y_,1),tf.argmax(y,1)) #计算预测值与真实值是否相等
#布尔值 转换成浮点数，并取平均值 得到的就是准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
with open('h2.pkl','rb') as f:
    h2_value=pickle.load(f)
with tf.Session() as sess:
    y_predict=sess.run(y_,feed_dict={h2:h2_value})
    print('y_predict:',y_predict)

