# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/7 0007 下午 3:12
 @Author  : Shanshan Wang
 @Version : Python3.5
"""
import numpy  as np

def get_vocabulary_size(X):
    # the 0th word represent the UNK words
    return max([max(x) for x in X])+1 # plus the 0th word
def fit_in_vocabulary(X,voc_size):
    return [[w for w in x if w<voc_size] for x in X]
def zero_pad(X,seq_len):
    return np.array([x[:seq_len-1]+[0]*max(seq_len-len(x),1) for x in X])
def batch_generator(X,Y,batch_size):
    '''
    Primitive batch generator
    :param X:
    :param Y:
    :param batch_size:
    :return:
    '''
    size=X.shape[0]
    X_copy=X.copy()
    Y_copy=Y.copy()
    indices=np.arange(size)
    np.random.shuffle(indices)
    X_copy=X_copy[indices]
    Y_copy=Y_copy[indices]
    i=0

    while True:
        if i+ batch_size<=size:
            yield  X_copy[i:i+batch_size],Y_copy[i:i+batch_size]
            i+=batch_size
        else:
            i=0
            indices=np.arange(size)
            np.random.shuffle(indices)
            X_copy=X_copy[indices]
            Y_copy=Y_copy[indices]
            continue

if __name__ == '__main__':
    # Test batch generator
    gen=batch_generator(np.array(['a', 'b', 'c', 'd']),np.array([1,2,3,4]),2)
    for _ in range(8):
        xx,yy=next(gen)
        print('xx:',xx)
        print('yy:',yy)

