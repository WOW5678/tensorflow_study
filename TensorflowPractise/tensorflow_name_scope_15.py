# -*- coding: utf-8 -*-
"""
 @Time    : 2018/10/18 0018 下午 6:20
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 了解tf的作用于name_scope的使用
"""
import tensorflow as tf
import os

# 只显示warning 和error
os.environ['TF_CNN_MIN_LOG_LEVEL']='2'

with tf.Session() as sess:
    with tf.variable_scope(name_or_scope='foo',initializer=tf.constant_initializer(3.0)) as foo:
        v=tf.get_variable(name='v',shape=[1])
        w=tf.get_variable(name='w',shape=[1])
        with tf.variable_scope(name_or_scope='bar'):
            l=tf.get_variable(name='h',shape=[1])
            with tf.variable_scope(name_or_scope=foo):
                h=tf.get_variable(name='h',shape=[1])
                g=v+w+l+h
    sess.run(tf.global_variables_initializer())
    print('\n{}---{}'.format(v.name,v.eval()))
    print('\n{}---{}'.format(w.name, w.eval()))
    print('\n{}---{}'.format(l.name, l.eval()))
    print('\n{}---{}'.format(h.name, h.eval()))
    print('\n{}---{}'.format(g.name, g.eval()))

# foo/v:0---[3.]
# foo/w:0---[3.]
# foo/bar/h:0---[3.]
# foo/h:0---[3.]
# foo/bar/foo/add_2:0---[12.]