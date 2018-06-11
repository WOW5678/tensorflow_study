# -*- coding:utf-8 -*-
#功能：变量的用法

import tensorflow as tf
state=tf.Variable(1,name='count')
#控制台输出的是 变量的名字和变量是第几个变量
print (state.name)

one=tf.constant(1)
new_value=tf.add(state,one)
update=tf.assign(state,new_value)
#只要定义了变量 这一步就必不可少
init=tf.initialize_all_variables()

with tf.Session() as sess:
    #定义了变量 也必须加上这一步
    sess.run(init)
    for _ in range(3):
        #如果要打印中间的变量 也必须加上session
        sess.run(update)
        print (sess.run(state))
