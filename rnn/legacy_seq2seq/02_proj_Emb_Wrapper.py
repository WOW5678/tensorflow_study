# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/17 0017 下午 3:56
 @Author  : Shanshan Wang
 @Version : Python3.5
"""
import matplotlib.pyplot as plt
from pprint import pprint
import re
from collections import Counter
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import *
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import  *

# sample training data
enc_sentence_length=10
dec_sentence_length=10
batch_size=4

input_batches=[
    ['Hi What is your name?', 'Nice to meet you!'],
    ['Which programming language do you use?', 'See you later.'],
    ['Where do you live?', 'What is your major?'],
    ['What do you want to drink?', 'What is your favorite beer?']
]
target_batches=[
    ['Hi this is Jaemin.', 'Nice to meet you too!'],
    ['I like Python.', 'Bye Bye.'],
    ['I live in Seoul, South Korea.', 'I study industrial engineering.'],
    ['Beer please!', 'Leffe brown!']
]

all_input_sentences=[]
for input_batch in input_batches:
    all_input_sentences.extend(input_batch)

all_target_sentences=[]
for target_batch in target_batches:
    all_target_sentences.extend(target_batch)

# NLP Helper function
def tokenizer(sentence):
    tokens=re.findall(r'[\w]+|[^\s\w]',sentence)
    return tokens

# Example
print(tokenizer('Hello world?? "sdfs%@#%'))

def build_vocab(sentences,is_target=False,max_vocab_size=None):
    word_counter = Counter()
    vocab = dict()
    reverse_vocab = dict()

    for sentence in sentences:
        tokens = tokenizer(sentence)
        word_counter.update(tokens)

    if max_vocab_size is None:
        max_vocab_size = len(word_counter)

    if is_target:
        vocab['_GO'] = 0
        vocab['_PAD'] = 1
        vocab_idx = 2
        for key, value in word_counter.most_common(max_vocab_size):
            vocab[key] = vocab_idx
            vocab_idx += 1
    else:
        vocab['_PAD'] = 0
        vocab_idx = 1
        for key, value in word_counter.most_common(max_vocab_size):
            vocab[key] = vocab_idx
            vocab_idx += 1

    for key, value in vocab.items():
        reverse_vocab[value] = key

    return vocab, reverse_vocab, max_vocab_size
#Example
# pprint(build_vocab(all_input_sentences))
# print('\n')
# pprint(build_vocab(all_target_sentences))

enc_vocab, enc_reverse_vocab, enc_vocab_size = build_vocab(all_input_sentences)
dec_vocab, dec_reverse_vocab, dec_vocab_size = build_vocab(all_target_sentences, is_target=True)

def token2idx(word, vocab):
    return vocab[word]

for token in tokenizer('Nice to meet you!'):
    print(token, token2idx(token, enc_vocab))

def sent2idx(sent, vocab=enc_vocab, max_sentence_length=enc_sentence_length, is_target=False):
    tokens = tokenizer(sent)
    current_length = len(tokens)
    pad_length = max_sentence_length - current_length
    if is_target:
        return [0] + [token2idx(token, vocab) for token in tokens] + [1] * pad_length
    else:
        return [token2idx(token, vocab) for token in tokens] + [0] * pad_length, current_length

# Enc Example
# print('Hi What is your name?')
# print(sent2idx('Hi What is your name?'))
#
# # Dec Example
# print('Hi this is Jaemin.')
# print(sent2idx('Hi this is Jaemin.', vocab=dec_vocab, max_sentence_length=dec_sentence_length, is_target=True))

def idx2token(idx, reverse_vocab):
    return reverse_vocab[idx]

def idx2sent(indices, reverse_vocab=dec_reverse_vocab):
    return " ".join([idx2token(idx, reverse_vocab) for idx in indices])

# Hyperparameters
n_epoch = 2000
n_enc_layer = 3
n_dec_layer = 3
hidden_size = 30

enc_emb_size = 50
dec_emb_size = 50

# Build Graph

tf.reset_default_graph()

enc_inputs = tf.placeholder(
    tf.int32,
    shape=[None, enc_sentence_length],
    name='input_sentences')

sequence_lengths = tf.placeholder(
    tf.int32,
    shape=[None],
    name='sentences_length')

dec_inputs = tf.placeholder(
    tf.int32,
    shape=[None, dec_sentence_length+1],
    name='output_sentences')

# batch major => time major
enc_inputs_t = tf.transpose(enc_inputs, [1,0])
dec_inputs_t = tf.transpose(dec_inputs, [1,0])

with tf.variable_scope('encoder'):
    enc_cell=tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    #直接将sequnce ids传入模型，也不用embedding,在此处再进行embedding
    enc_cell=EmbeddingWrapper(enc_cell,enc_vocab_size+1,enc_emb_size)

    # static_rnn的inputs是一个列表，每个元素的shape:[batch_size,embedding_size]
    enc_outputs,enc_last_state=tf.contrib.rnn.static_rnn(
        cell=enc_cell,
        inputs=tf.unstack(enc_inputs_t),
        sequence_length=sequence_lengths,
        dtype=tf.float32
    )

dec_ouputs=[]
dec_predictions=[]
with tf.variable_scope('decoder') as scope:
    dec_cell=tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    #此类就是在 cell 前 加了一层embedding
    #通过这个包装器，可以将模型的输入设计为：[batch_size,1],输出为:[ouput,state]
    dec_cell=EmbeddingWrapper(dec_cell,dec_vocab_size+2,dec_emb_size)
    dec_cell=OutputProjectionWrapper(dec_cell,dec_vocab_size+2)

    for i in range(dec_sentence_length+1):
        if i==0:
            input_=dec_inputs_t[i]
            print('input_:',input)
            state=enc_last_state
        else:
            scope.reuse_variables()
            input_=dec_prediction

        # dec_output:[batch_size,dec_vocab_size+2]
        # state:[batch_size,hidden_size]
        dec_output,state=dec_cell(input_,state)

        # dec_prediction:[batch_size,1]
        dec_prediction=tf.argmax(dec_output,axis=1)
        dec_ouputs.append(dec_output)
        dec_predictions.append(dec_prediction)
# Predictions:[batch_size,dec_sentence_length]
predictions=tf.transpose(tf.stack(dec_predictions),[1,0])

# Labels & logits:[dec_sentence_length+1,batch_size,dec_vocab_size+2]
labels=tf.one_hot(dec_inputs_t,dec_vocab_size+2)
logits=tf.stack(dec_ouputs)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))
# Training ops
training_op=tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)

# Run Graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_history = []
    for epoch in range(n_epoch):
        all_preds = []
        epoch_loss = 0
        for input_batch, target_batch in zip(input_batches, target_batches):
            input_token_indices = []
            target_token_indices = []
            sentence_lengths = []

            for input_sent in input_batch:
                input_sent, sent_len = sent2idx(input_sent)
                input_token_indices.append(input_sent)
                sentence_lengths.append(sent_len)

            for target_sent in target_batch:
                target_token_indices.append(
                    sent2idx(target_sent,
                             vocab=dec_vocab, max_sentence_length=dec_sentence_length, is_target=True))

            # Evaluate three operations in the graph
            # => predictions, loss, training_op(optimazier)
            batch_preds, batch_loss, _ = sess.run(
                [predictions, loss, training_op],
                feed_dict={
                    enc_inputs: input_token_indices,
                    sequence_lengths: sentence_lengths,
                    dec_inputs: target_token_indices
                })
            loss_history.append(batch_loss)
            epoch_loss += batch_loss
            all_preds.append(batch_preds)

        # Logging every 400 epochs
        if epoch % 400 == 0:
            print('Epoch', epoch)
            for input_batch, target_batch, batch_preds in zip(input_batches, target_batches, all_preds):
                for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
                    print('\t', input_sent)
                    print('\t => ', idx2sent(pred, reverse_vocab=dec_reverse_vocab))
                    print('\tCorrent answer:', target_sent)
            print('\tepoch loss: {:.2f}\n'.format(epoch_loss))

# Learning Curve
plt.figure(figsize=(20, 10))
plt.scatter(range(n_epoch * batch_size), loss_history)
plt.title('Learning Curve')
plt.xlabel('Global step')
plt.ylabel('Loss')
plt.show()
