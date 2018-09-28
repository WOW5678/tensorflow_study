# -*- coding: utf-8 -*-
"""
 @Time    : 2018/9/26 0026 上午 9:13
 @Author  : Shanshan Wang
 @Version : Python3.5
 function: implement a basic seq2seq model.
 模型将返回一个对字符排序后的“单词”
"""
import random
random.seed(1001)
import numpy as np
np.random.seed(1001)
from tensorflow import set_random_seed
set_random_seed(1001)
from distutils.version import  LooseVersion
import tensorflow as tf
from tensorflow.python.layers.core import Dense



# check tensorflow version
assert LooseVersion(tf.__version__)>=LooseVersion('1.1'),'Please use tensorlfow version 1.1 or newer'
#1.9.0
print('Tensorflow Version:{}'.format(tf.__version__))

#数据加载
import numpy as np
import time
import tensorflow as tf
with open('temp/letters_source.txt') as f:
    source_data=f.read()
with open('temp/letters_target.txt') as f:
    target_data=f.read()

# 数据预览
print(source_data.split('\n')[:10])
print(target_data.split('\n')[:10 ])

# 数据预处理
def extract_character_vocab(data):
    '''
    构造映射表
    :param data:
    :return:
    '''
    special_words=['<PAD>','<UNK>','<GO>','<EOS>']
    set_words=list(set([character for line in data.split('\n') for character in line]))
    #这里要把四个特殊字符添加到字典中
    int_to_vocab={idx:word for idx,word in enumerate(special_words+set_words)}
    vocab_to_int={word:idx for idx,word in int_to_vocab.items()}
    return int_to_vocab,vocab_to_int

# 构造映射表
source_int_to_letter,source_letter_to_int=extract_character_vocab(source_data)
target_int_to_letter,target_letter_to_int=extract_character_vocab(target_data)

#对字符进行转换，将字符序列转换成id序列
source_int=[[source_letter_to_int.get(letter,source_letter_to_int['<UNK>']) for letter in line] for line in source_data.split('\n')]
target_int=[[target_letter_to_int.get(letter,target_letter_to_int['<UNK>']) for letter in line]+[target_letter_to_int['<EOS>']] for line in target_data.split('\n')]

print('source_int[:10]',source_int[:10])
print('target_int[:10]',target_int[:10])

# 构建模型
#输入层
def get_inputs():
    '''
    模型输入tensor
    :return:
    '''
    inputs=tf.placeholder(tf.int32,[None,None],name='inputs')
    targets=tf.placeholder(tf.int32,[None,None],name='targets')
    learning_rate=tf.placeholder(tf.float32,name='learning_rate')

    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length=tf.placeholder(tf.int32,(None,),name='target_sequence_length')
    max_target_sequence_length=tf.reduce_max(target_sequence_length,name='max_target_sequence_length')
    source_sequence_length=tf.placeholder(tf.int32,(None,),name='source_sequence_length')
    return inputs,targets,learning_rate,target_sequence_length,max_target_sequence_length,source_sequence_length

# Encoder
def get_encoder_layer(input_data,rnn_size,num_layers,source_sequence_length,source_vocab_size,encoding_embedding_size):
    '''
    构造embedding层
    :param input_data:输入tensor
    :param rnn_size:rnn隐层节点的个数
    :param num_layers:堆叠rnn cell的数量
    :param source_sequence_length:源数据的序列长度
    :param source_vocab_size:源数据的词典大小
    :param encoding_embedding_size:embedding的大小
    :return:
    '''
    # Encoder embedding
    encoder_embed_input=tf.contrib.layers.embed_sequence(input_data,source_vocab_size,encoding_embedding_size)
    print('encoder_embed_input:',encoder_embed_input)

    # RNN SIZE
    def get_lstm_cell(rnn_size):
        lstm_cell=tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1,0.1,seed=2))
        return lstm_cell
    cell=tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
    # A length T list of inputs, each a `Tensor` of shape
    #  `[batch_size, input_size]
    encoder_output,encoder_state=tf.nn.dynamic_rnn(cell,encoder_embed_input,sequence_length=source_sequence_length,dtype=tf.float32)
    return encoder_output,encoder_state

#Decoder
#对target 数据进行预处理
def process_decoder_input(data,vocab_to_int,batch_size):
    '''
    补充<GO>，并移除最后一个字符
    :param data: target_data
    :param vocab_to_int: target_letter_to_int
    :param batch_size:batch_size
    :return:
    '''
    # cut 掉最后一个字符
    ending=tf.strided_slice(data,[0,0],[batch_size,-1],strides=[1,1])
    decoder_input=tf.concat([tf.fill([batch_size,1],vocab_to_int['<GO>']),ending],1)
    return decoder_input

def decoding_layer(target_letter_to_int,decoding_embedding_size,num_layers,rnn_size,
                   target_sequence_length,max_target_sequence_length,encoder_state,decoder_input):
    '''

    :param target_letter_to_int:target数据的映射表
    :param decoding_embedding_size:embed向量大小
    :param num_layers:堆叠RNN单元数量
    :param rnn_size:RNN单元的隐层结点数量
    :param target_sequence_length:target数据序列长度
    :param max_target_sequence_length:target序列最大长度
    :param encoder_state:encoder端编码的状态向量
    :param decoder_input:decoder端输入
    :return:
    '''
    #1. Embedding层
    target_vocab_size=len(target_letter_to_int)
    decoder_embeddings=tf.Variable(tf.random_uniform([target_vocab_size,decoding_embedding_size]))
    decoder_embed_input=tf.nn.embedding_lookup(decoder_embeddings,decoder_input)

    #2 构造Decoder中的RNN单元
    def get_decoder_cell(rnn_size):
        decoder_cell=tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1,0.1,seed=2))
        return decoder_cell
    cell=tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])

    # 3.output 全连接层
    output_layer=Dense(target_vocab_size,
                       kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))
    # 4. Training Decoder

    ''''
    ## tf.contrib.seq2seq.TrainingHelper：##
    Decoder端用来训练的函数。 
    这个函数不会把t-1阶段的输出作为t阶段的输入，而是把target中的真实值直接输入给RNN。
    主要参数是inputs和sequence_length。返回helper对象，可以作为BasicDecoder函数的参数。
    ## tf.contrib.seq2seq.GreedyEmbeddingHelper：##
    它和TrainingHelper的区别在于它会把t-1下的输出进行embedding后再输入给RNN。
    '''
    with tf.variable_scope('decoder'):
        # 得到helper对象
        training_helper=tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                          sequence_length=target_sequence_length,
                                                          time_major=False)
        # 构造decoder
        training_decoder=tf.contrib.seq2seq.BasicDecoder(cell,
                                                         training_helper,
                                                         encoder_state,
                                                         output_layer)
        training_decoder_output,_,final_sequence_lengths=tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                    impute_finished=True,
                                                                    maximum_iterations=max_target_sequence_length)

    # 5. Predicting decoder
    # 与 training共享参数
    with tf.variable_scope('decoder',reuse=True):
        # 创建一个常量tensor 并复制为batch_size的大小
        start_tokens=tf.tile(tf.constant([target_letter_to_int['<GO>']],dtype=tf.int32),[batch_size],
                             name='start_tokens')
        predicting_helper=tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                   start_tokens,
                                                                   target_letter_to_int['<EOS>'])
        predicting_decoder=tf.contrib.seq2seq.BasicDecoder(cell,predicting_helper,
                                                           encoder_state,
                                                           output_layer)
        predicting_decoder_output,_,predicting_final_sequence_length=tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                      impute_finished=True,
                                                                      maximum_iterations=max_target_sequence_length)
    return training_decoder_output,predicting_decoder_output

# Seq2seq模型
# 将encoder与decoder这两个部分连接起来，构造seq2seq模型

def seq2seq_model(input_data,targets,lr,target_sequence_length,
                  max_target_sequence_length,source_sequence_length,
                  source_vocab_size,target_vocab_size,
                  encoder_embedding_size,decoder_embedding_size,
                  rnn_size,num_layers):
    # 获取encoder的状态输出
    _,encoder_state=get_encoder_layer(input_data,
                                      rnn_size,
                                      num_layers,
                                      source_sequence_length,
                                      source_vocab_size,
                                      encoding_embedding_size)
    # 预处理后的decoder输入
    decoder_input=process_decoder_input(targets,target_letter_to_int,batch_size)
    #将状态向量与输入传递给decoder
    training_decoder_output,predicting_decoder_output=decoding_layer(target_letter_to_int,
                                                                     decoding_embedding_size,
                                                                     num_layers,
                                                                     rnn_size,
                                                                     target_sequence_length,
                                                                     max_target_sequence_length,
                                                                     encoder_state,
                                                                     decoder_input)
    return training_decoder_output,predicting_decoder_output
# 超参数
epochs=60
batch_size=128
rnn_size=50
num_layers=2
encoding_embedding_size=15
decoding_embedding_size=15
learning_rate=0.001

# 构造Graph
train_graph=tf.Graph()
with train_graph.as_default():
    # 获取模型的输入
    input_data,targets,lr,target_sequence_length,max_target_sequence_length, source_sequence_length = get_inputs()
    training_decoder_output,predicting_decoder_output=seq2seq_model(input_data,
                                                                    targets,
                                                                    lr,
                                                                    target_sequence_length,
                                                                    max_target_sequence_length,
                                                                    source_sequence_length,
                                                                    len(source_letter_to_int),
                                                                    len(target_letter_to_int),
                                                                    encoding_embedding_size,
                                                                    decoding_embedding_size,
                                                                    rnn_size,
                                                                    num_layers)
    #tf.identity是返回一个一模一样新的tensor的op，这会增加一个新节点到gragh中
    #training_logits:shape=(batch_size, ?, 30) 即[batch_size, sequence_length, num_decoder_symbols]                                                     )
    training_logits=tf.identity(training_decoder_output.rnn_output,'logits')
    #predicting_logits 后面没再使用，这里复制出一份的目的是？？？？
    predicting_logits=tf.identity(predicting_decoder_output.sample_id,name='predictions')
    masks=tf.sequence_mask(target_sequence_length,max_target_sequence_length,dtype=tf.float32,name='mask')

    with tf.name_scope('optimization'):
        #Loss function
        print('traning_logits:',training_logits)
        print('targets:',targets)
        #targets:[batch_size, sequence_length]
        cost=tf.contrib.seq2seq.sequence_loss(training_logits,targets,masks)
        #optimizer
        optimizer=tf.train.AdamOptimizer(lr)
        # Gradient clipping 基于定义的min和max对tensor数据进行截断操作，
        #目的是为了应对梯度爆炸或者梯度消失的问题
        gradients=optimizer.compute_gradients(cost)
        capped_gradients=[(tf.clip_by_value(grad,-5,5),var) for grad,var in gradients if grad is not None]
        train_op=optimizer.apply_gradients(capped_gradients)

# Batches
def pad_sentence_batch(sentence_batch,pad_in):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
    :param sentence_batch:
    :param pad_in:<PAD>对应索引号
    :return:
    '''
    max_sentence=max([len(sentence)for sentence in sentence_batch])
    return [sentence+[pad_in]*(max_sentence-len(sentence)) for sentence in sentence_batch]

def get_batches(targets,sources,batch_size,source_pad_in,target_pad_int):
    '''
    定义生成器，用来获取batch
    :param targets:
    :param sources:
    :param batch_size:
    :param source_pad_in:<PAD>对应索引号
    :param target_pad_int:<PAD>对应索引号
    :return:
    '''
    for batch_i in range(0,len(sources)//batch_size):
        start_i=batch_i*batch_size
        sources_batch=sources[start_i:start_i+batch_size]
        target_batch=targets[start_i:start_i+batch_size]

        #补全序列
        pad_sources_batch=np.array(pad_sentence_batch(sources_batch,source_pad_in))
        pad_targets_batch=np.array(pad_sentence_batch(target_batch,target_pad_int))

        # 记录每条记录的长度
        pad_targets_lengths=[]
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))
        pad_source_lengths=[]
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_targets_batch,pad_sources_batch,pad_targets_lengths,pad_source_lengths

# Training
#将数据集分割为train 和validation
train_source=source_int[batch_size:]
train_target=target_int[batch_size:]
# 留出一个batch进行验证
valid_source=source_int[:batch_size]
valid_target=target_int[:batch_size]
(valid_target_batch,valid_source_batch,valid_target_lengths,valid_source_lengths)=next(
  get_batches(valid_target,valid_source,batch_size,source_letter_to_int['<PAD>'],target_letter_to_int['<PAD>'])
)

display_step=50 #每隔50轮输出loss
check_points='trained_model.ckpt'
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(1,epochs+1):
        for batch_i,(targets_batch,sources_batch,targets_lengths,sources_lengths) in enumerate(
            get_batches(train_target,train_source,batch_size,source_letter_to_int['<PAD>'],
                           target_letter_to_int['<PAD>'])
        ):
         _,loss=sess.run([train_op,cost],
                          {input_data:sources_batch,targets:targets_batch,lr:learning_rate,
                           target_sequence_length:targets_lengths,
                           source_sequence_length:sources_lengths})
         if batch_i %display_step==0:
             # 计算validation loss
            validation_loss=sess.run([cost],{input_data:valid_source_batch,
                                             targets:valid_target_batch,
                                             lr:learning_rate,
                                             target_sequence_length:valid_target_lengths,
                                             source_sequence_length:valid_source_lengths})

            print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                   .format(epoch_i,
                           epochs,
                           batch_i,
                           len(train_source) // batch_size,
                           loss,
                           validation_loss[0]))
        # 保存模型
    saver = tf.train.Saver()
    saver.save(sess, check_points)
    print('Model Trained and Saved')





