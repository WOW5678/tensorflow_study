# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/10 0010 下午 7:21
 @Author  : Shanshan Wang
 @Version : Python3.5
"""
from train import *
import tensorflow as tf

saver=tf.train.Saver()

# Calculate alpha coefficients for the first test example

with tf.Session() as sess:
    saver.restore(sess,MODEL_PATH)
    x_batch_test,y_batch_test=x_test[:1],y_test[:1]
    seq_len_test=np.array([list(x).index(0)+1 for x in x_batch_test])
    print('seq_len_test:',seq_len_test)
    alphas_test=sess.run([alphas],feed_dict={batch_ph:x_batch_test,target_ph:y_batch_test,
                                             seq_len_ph:seq_len_test,keep_prob_ph:1.0})

alphas_values=alphas_test[0][0]
print('alphas_values:',alphas_values)
# Build correct mapping from word to index and inverse
word_index=imdb.get_word_index()
print('word_index:',word_index)
word_index={word:index+INDEX_FROM for word,index in word_index.items()}
word_index[':PAD:']=0
word_index[':START:']=1
word_index[':UNK:']=2
index_word={value:key for key,value in word_index.items()}

# Represent the sample by words rather than indices
words=list(map(index_word.get,x_batch_test[0]))

# Save visualization as HTML
with open('visualization.html','w') as html_file:
    # 对alphas值进行归一化处理
    for word,alpha in zip(words,alphas_values/alphas_values.max()):
        if word==':START:':
            continue
        elif word==':PAD:':
            break
        html_file.write('<font style="background: rgba(255, 255, 0, %f)">%s</font>\n' % (alpha, word))

print('\nOpen visualization.html to checkout the attention coefficients visualization.')
