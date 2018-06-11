# -*- coding:utf-8 -*-
#功能：介绍dropout的使用
import  tensorflow  as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

#加载数据集
digits=load_digits()
x=digits.data
y=digits.target
#将标签转化成数值型
y=LabelBinarizer().fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#增加层
def addLayer(inputs,in_size,out_size,layer_name,activaiton_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biaes=tf.Variable(tf.zeros([1,out_size])+0.1,)
    wx_plus_b=tf.matmul(inputs,Weights)+biaes
    wx_plus_b=tf.nn.dropout(wx_plus_b,keep_prob)
    if activaiton_function is None:
        output=wx_plus_b
    else:
        output=activaiton_function(wx_plus_b)
    tf.histogram_summary(layer_name+'/output',output)
    return output

keep_prob=tf.placeholder(tf.float32)
xs=tf.placeholder(tf.float32,[None,64])
ys=tf.placeholder(tf.float32,[None,10])

#增加输出层
l1=addLayer(xs,64,50,'l1',activaiton_function=tf.nn.tanh)
prediction=addLayer(l1,50,10,'l2',activaiton_function=tf.nn.softmax)

#损失函数
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
tf.scalar_summary('loss',cross_entropy)
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess=tf.Session()
merged=tf.merge_all_summaries()
train_writer=tf.train.SummaryWriter('logs/train',sess.graph)
test_writer=tf.train.SummaryWriter('logs/test',sess.graph)

sess.run(tf.initialize_all_variables())

#训练
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_train,ys:y_train,keep_prob:0.5})
    if i%100==0:
        train_result=sess.run(merged,feed_dict={xs:x_train,ys:y_train,keep_prob:1})
        test_result=sess.run(merged,feed_dict={xs:x_test,ys:y_test,keep_prob:1})
        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)
