# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/17 0017 下午 6:19
 @Author  : Shanshan Wang
 @Version : Python3.5
"""
import matplotlib.pyplot as plt
from pprint import pprint
import re
from collections import Counter
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import *
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import *
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import *

# Sample training data
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

# Example
#['Hi What is your name?', 'Nice to meet you!',...]
#print(all_input_sentences)

# NLP Helper function
def tokenizer(sentence):
    tokens=re.findall(r"[\w]+|[^\s\w]",sentence)
    return tokens
# Example
#['Hello', 'world', '?', '?', '"', 'sdfs', '%', '@', '#', '%']
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

# Example
# pprint(build_vocab(all_input_sentences))
# print('\n')
# pprint(build_vocab(all_target_sentences,is_target=True))
enc_vocab, enc_reverse_vocab, enc_vocab_size = build_vocab(all_input_sentences)
dec_vocab, dec_reverse_vocab, dec_vocab_size = build_vocab(all_target_sentences, is_target=True)

def token2idx(word,vocab):
    return vocab[word]
# for token in tokenizer('Nice to meet you！'):
#     print(token,token2idx(token,enc_vocab))

def sent2idx(sent,vocab=enc_vocab,max_sentence_length=enc_sentence_length,is_target=False):
    tokens=tokenizer(sent)
    current_length = len(tokens)
    pad_length = max_sentence_length - current_length
    if is_target:
        return [0] + [token2idx(token, vocab) for token in tokens] + [1] * pad_length
    else:
        return [token2idx(token, vocab) for token in tokens] + [0] * pad_length, current_length

# Enc Example
print('Hi What is your name?')
#([18, 3, 6, 5, 9, 1, 0, 0, 0, 0], 6)
print(sent2idx('Hi What is your name?'))

# Dec Example
print('Hi this is Jaemin.')
#[0, 18, 6, 23, 7, 2, 1, 1, 1, 1, 1]
print(sent2idx('Hi this is Jaemin.', vocab=dec_vocab, max_sentence_length=dec_sentence_length, is_target=True))

def idx2token(idx,reverse_vocab):
    return reverse_vocab[idx]
def idx2sent(indices,reverse_vocab=dec_reverse_vocab):
    return ' '.join([idx2token(idx,reverse_vocab) for idx in indices])

# Hyperparameters
n_epochs=2000
n_enc_layer=3
n_dec_layer=3
hidden_size=30

enc_emb_size=50
dec_emb_size=50

# Build Graph

# https://github.com/tensorflow/tensorflow/blob/r1.0/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py#L80# https:
# Couldn't import '_' prefixed methods... So just manually copied.
def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
    '''
    Get a loop_function that extracts the previous symbol and embeds it
    :param embedding: embedding tensor for symbols
    :param output_projection: None or a pair (W,B).If provided,each fed previous
    #output will first be multiplied by W and added B.
    :param update_embedding: Boolean,if false,the gradients will not propagate
    through the embeddings.
    :return: A loop function
    '''
    def loop_function(prev,_):
        if output_projection is not None:
            pre=tf.nn.xw_plus_b(prev,output_projection[0],output_projection[1])
        prev_symbol=tf.argmax(prev,1)
        #Not that gradients will not propagate through the second parameter
        # of embedding_lookup
        emb_prev=tf.nn.embedding_lookup(embedding,prev_symbol)
        if not update_embedding:
            emb_prev=tf.stop_gradient(emb_prev)
        return emb_prev
    return loop_function

tf.reset_default_graph()
enc_inputs=tf.placeholder(tf.int32,[None,enc_sentence_length],name='input_sentences')
sequence_lenghts=tf.placeholder(tf.int32,[None],name='sentences_length')
dec_inputs=tf.placeholder(tf.int32,[None,dec_sentence_length+1],name='output_senteces')

# batch_major => time_major
# 这么处理的目的是为了符合basic_rnn模型的输入
enc_inputs_t = tf.transpose(enc_inputs, [1,0])
dec_inputs_t = tf.transpose(dec_inputs, [1,0])

with tf.device('/cpu:0'):
    dec_Wemb=tf.get_variable('dec_word_emb',
             initializer=tf.random_uniform([dec_vocab_size+2,dec_emb_size]))

with tf.variable_scope('encoder'):
    enc_cell=tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    # 相当于在cell之前加上了一个embedding层
    enc_cell=EmbeddingWrapper(enc_cell,enc_vocab_size+1,enc_emb_size)

    # enc_outputs:[enc_sent_Length,batch_size,embedding_size]
    # enc_last_state:[batch_size,hidden_size]
    enc_outputs,enc_last_state=tf.contrib.rnn.static_rnn(
        cell=enc_cell,
        inputs=tf.unstack(enc_inputs_t),
        sequence_length=sequence_lenghts,
        dtype=tf.float32
    )


top_states=[]
with tf.variable_scope('attention'):
    #top_states:enc_sent_len个[batch_size,1,embedding_size]
    #enc_outputs has length of enc_sent_len(=max_enc_len)
    for enc_output in enc_outputs:
        top_states.append(tf.reshape(enc_output,[-1,1,enc_cell.output_size]))
    # attention_states:[batch_size,enc_sent_len,embedding_size]
    attention_states=tf.concat(top_states,1)

dec_predicitons=[]

with tf.variable_scope('decoder'):
    dec_cell=tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    dec_cell=OutputProjectionWrapper(dec_cell,dec_vocab_size+2)

    # EmbeddingWrapper & tf.unstack(dec_inputs_t) raises dimension error
    # dec_emb_inputs;[dec_sequence_Len,batch_size,embedding_size]
    dec_emb_inputs=tf.nn.embedding_lookup(dec_Wemb,dec_inputs_t)
    #print('dec_emb_inputs:',dec_emb_inputs)

    # dec_outputs:[dec_sent_len+1，batch_size,hidden_size]
    dec_outputs,dec_last_state=attention_decoder(
        decoder_inputs=tf.unstack(dec_emb_inputs),
        initial_state=enc_last_state,
        attention_states=attention_states,
        cell=dec_cell,
        loop_function=_extract_argmax_and_embed(dec_Wemb)
    )
# Predictions:[batch_size,dec_sentence_length+1]
#print('tf.argmax(tf.stack(dec_outputs),axis=-1):',tf.argmax(tf.stack(dec_outputs),axis=-1))
predicitons=tf.transpose(tf.argmax(tf.stack(dec_outputs),axis=-1),[1,0])
#labels & logits:[]
labels=tf.one_hot(dec_inputs_t,dec_vocab_size+2)
logits=tf.stack(dec_outputs)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=labels,logits=logits
))
training_op=tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)

# RUN Graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_history=[]

    for epoch in range(n_epochs):
        all_preds=[]
        epoch_loss=0.0
        for input_batch,target_batch in zip(input_batches,target_batches):
            input_token_indices=[]
            target_token_indices=[]
            sentence_lengths=[]

            for input_sent in input_batch:
                input_sent,sent_len=sent2idx(input_sent)
                input_token_indices.append(input_sent)
                sentence_lengths.append(sent_len)
            for target_sent in target_batch:
                target_token_indices.append(
                    sent2idx(target_sent,vocab=dec_vocab,max_sentence_length=dec_sentence_length,is_target=True)
                )

            # Evaluate three operations in the graph
            # => predicitons,loss,training_op(optimizer)
            batch_preds,batch_loss,_=sess.run([predicitons,loss,training_op],
                                        feed_dict={
                                            enc_inputs:input_token_indices,
                                            sequence_lenghts:sentence_lengths,
                                            dec_inputs:target_token_indices
                                        })
            loss_history.append(batch_loss)
            epoch_loss+=batch_loss
            all_preds.append(batch_preds)
        if epoch%400==0 or epoch==n_epochs-1:
            print('epoch:',epoch)
            for input_batch, target_batch, batch_preds in zip(input_batches, target_batches, all_preds):
                for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
                    print('\t', input_sent)
                    print('\t => ', idx2sent(pred, reverse_vocab=dec_reverse_vocab))
                    print('\tCorrent answer:', target_sent)
            print('\tepoch loss: {:.2f}\n'.format(epoch_loss))

# Learning Curve
plt.figure(figsize=(20,10))
plt.scatter(range(n_epochs*batch_size),loss_history)
plt.title('Learning Curve')
plt.xlabel('Global step')
plt.ylabel('Loss')
plt.show()

