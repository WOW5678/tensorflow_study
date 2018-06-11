# -*- coding:utf-8 -*-
#功能：使用tensorflow实现一个最简单的word2vec模型  skip-gram模型 使用负采样的方法进行训练

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function


import collections

import math

import os

import random

#解压缩文件需要使用的模块
import zipfile



import numpy as np

from six.moves import urllib

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf



# Step 1: Download the data.

url = 'http://mattmahoney.net/dc/'





def maybe_download(filename, expected_bytes):

  """Download a file if not present, and make sure it's the right size."""

  if not os.path.exists(filename):

    filename, _ = urllib.request.urlretrieve(url + filename, filename)

  statinfo = os.stat(filename)

  #判断是否下载完整
  if statinfo.st_size == expected_bytes:

    print('Found and verified', filename)

  else:

    print(statinfo.st_size)

    raise Exception(

        'Failed to verify ' + filename + '. Can you get to it with a browser?')

  #返回数据集的名称
  return filename



filename = maybe_download('text8.zip', 31344016)





# Read the data into a list of strings.

def read_data(filename):

  """Extract the first file enclosed in a zip file as a list of words"""
  """Extract the first file enclosed in a zip file as a list of words"""

  with zipfile.ZipFile(filename) as f:
    #文件中可能包含多个文件，选择第一个文件进行解压缩和使用 空格分割

    data = tf.compat.as_str(f.read(f.namelist()[0])).split()

  #返回第一个文件中的内容组成列表
  print (data)
  return data



words = read_data(filename)
#输出文件内容的长度 即单词个数
print('Data size', len(words))



# Step 2: Build the dictionary and replace rare words with UNK token.
#创建字典，并将稀缺的单词使用unk进行取代
vocabulary_size = 50000


def build_dataset(words):

  count = [['UNK', -1]]
  #collections.Counter是字典的一个子类 可以快速统计单词出现的次数 most_comon函数选择频率最高的n个单词
  #选择出现频率最高的前50000-1个单词加入到count中  因为count中第一个单词的位置留给了UNK
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

  dictionary = dict()
  #count列表中第一维为单词 第二维为出现的次数
  for word, _ in count:
    #为count中的每个单词建立索引 将每个单词的索引存储在dictionary中
    #dictionary字典存放每个单词的标号（index）
    dictionary[word] = len(dictionary)

  data = list()
  #unk_count记录着UNK的个数
  unk_count = 0

  #为每个单词赋予索引值  并记录UNK的个数
  for word in words:
    #dictionary是一个字典（仅保存常见的单词），key为单词，value为单词出现的次数
    if word in dictionary:

      index = dictionary[word]

    else:

      index = 0  # dictionary['UNK']

      unk_count += 1

    #将索引值添加到data列表中  data中存储着出现次数 对应着words中的单词
    data.append(index)

  #更新UNK的个数 -1--->unk_count
  count[0][1] = unk_count

  #将字典中单词和单词的索引进行反转 并压缩到一个新的字典中
  #即revese_dictionary字典中key为单词出现的次数, 值为单词
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

  return data, count, dictionary, reverse_dictionary



data, count, dictionary, reverse_dictionary = build_dataset(words)

del words  # Hint to reduce memory.

#输出最常见的5个单词  其中UNK放在第一位
print('Most common words (+UNK)', count[:5])

#输出前10个样本数据  并且每个样本数据为[单词出现次数，单词]
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])



data_index = 0





# Step 3: Function to generate a training batch for the skip-gram model.

def generate_batch(batch_size, num_skips, skip_window):

  global data_index

  assert batch_size % num_skips == 0

  assert num_skips <= 2 * skip_window
  #skip-gram模型的输入有两个 一个是上下文 另一个是目标单词

  #batch为一个batch_size长度的列表
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)

  #目标单词 label的格式为·[[1]
  #                     [1]]
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  #跨度
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]

  #collections.deque类似于队列 但是左右两边都可以进行删除和添加
  buffer = collections.deque(maxlen=span)

  for _ in range(span):
    #将每个跨度的索引添加到buffer中

    buffer.append(data[data_index])

    data_index = (data_index + 1) % len(data)

  for i in range(batch_size // num_skips):

    target = skip_window  # target label at the center of the buffer

    targets_to_avoid = [skip_window]

    for j in range(num_skips):

      while target in targets_to_avoid:

        target = random.randint(0, span - 1)

      targets_to_avoid.append(target)

      batch[i * num_skips + j] = buffer[skip_window]

      labels[i * num_skips + j, 0] = buffer[target]

    buffer.append(data[data_index])

    data_index = (data_index + 1) % len(data)

  return batch, labels



batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

for i in range(8):

  print(batch[i], reverse_dictionary[batch[i]],

        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])



# Step 4: Build and train a skip-gram model.



batch_size = 128

embedding_size = 128  # Dimension of the embedding vector.

skip_window = 1       # How many words to consider left and right.

num_skips = 2         # How many times to reuse an input to generate a label.


# We pick a random validation set to sample nearest neighbors. Here we limit the

# validation samples to the words that have a low numeric ID, which by

# construction are also the most frequent.

valid_size = 16     # Random set of words to evaluate similarity on.

valid_window = 100  # Only pick dev samples in the head of the distribution.

valid_examples = np.random.choice(valid_window, valid_size, replace=False)

num_sampled = 64    # Number of negative examples to sample.



graph = tf.Graph()



with graph.as_default():



  # Input data.

  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])

  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)



  # Ops and variables pinned to the CPU because of missing GPU implementation

  with tf.device('/cpu:0'):

    # Look up embeddings for inputs.

    embeddings = tf.Variable(

        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    #tf.nn.embedding_lookup函数的作用是用来检索某几行的
    #功能类似于：embedding_lookup函数中检索行的 params张。行为是类似于使用与中 numpy 的数组索引
    #matrix = np.random.random([1024, 64])  # 64-dimensional embeddings
    #ids = np.array([0, 5, 17, 33])
    #print matrix[ids]  # prints a matrix of shape [4, 64]


    embed = tf.nn.embedding_lookup(embeddings, train_inputs)



    # Construct the variables for the NCE loss

    nce_weights = tf.Variable(
         #用tf.truncated_normal生成正太分布的数据，作为W的初始值
        tf.truncated_normal([vocabulary_size, embedding_size],

                            stddev=1.0 / math.sqrt(embedding_size)))
    #始化b为可变的0矩阵
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))



  # Compute the average NCE loss for the batch.

  # tf.nce_loss automatically draws a new sample of the negative labels each

  # time we evaluate the loss.

  loss = tf.reduce_mean(

      tf.nn.nce_loss(weights=nce_weights,

                     biases=nce_biases,

                     labels=train_labels,

                     inputs=embed,

                     num_sampled=num_sampled,

                     num_classes=vocabulary_size))



  # Construct the SGD optimizer using a learning rate of 1.0.

  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)



  # Compute the cosine similarity between minibatch examples and all embeddings.

  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

  normalized_embeddings = embeddings / norm

  valid_embeddings = tf.nn.embedding_lookup(

      normalized_embeddings, valid_dataset)

  similarity = tf.matmul(

      valid_embeddings, normalized_embeddings, transpose_b=True)



  # Add variable initializer.

  #init = tf.global_variables_initializer()
  init=tf.initialize_all_variables()



# Step 5: Begin training.

num_steps = 100001



with tf.Session(graph=graph) as session:

  # We must initialize all variables before we use them.

  init.run()

  print("Initialized")



  average_loss = 0

  for step in xrange(num_steps):

    batch_inputs, batch_labels = generate_batch(

        batch_size, num_skips, skip_window)

    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}



    # We perform one update step by evaluating the optimizer op (including it

    # in the list of returned values for session.run()

    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)

    average_loss += loss_val



    if step % 2000 == 0:

      if step > 0:

        average_loss /= 2000

      # The average loss is an estimate of the loss over the last 2000 batches.

      print("Average loss at step ", step, ": ", average_loss)

      average_loss = 0



    # Note that this is expensive (~20% slowdown if computed every 500 steps)

    if step % 10000 == 0:

      sim = similarity.eval()

      for i in xrange(valid_size):

        valid_word = reverse_dictionary[valid_examples[i]]

        top_k = 8  # number of nearest neighbors

        nearest = (-sim[i, :]).argsort()[1:top_k + 1]

        log_str = "Nearest to %s:" % valid_word

        for k in xrange(top_k):

          close_word = reverse_dictionary[nearest[k]]

          log_str = "%s %s," % (log_str, close_word)

        print(log_str)

  final_embeddings = normalized_embeddings.eval()
  print (final_embeddings)



#Step 6: Visualize the embeddings.


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):

  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"

  plt.figure(figsize=(18, 18))  # in inches

  for i, label in enumerate(labels):

    x, y = low_dim_embs[i, :]

    plt.scatter(x, y)

    plt.annotate(label,

                 xy=(x, y),

                 xytext=(5, 2),

                 textcoords='offset points',

                 ha='right',

                 va='bottom')



  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE

  import matplotlib.pyplot as plt



  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

  plot_only = 500

  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])

  labels = [reverse_dictionary[i] for i in xrange(plot_only)]

  plot_with_labels(low_dim_embs, labels)



except ImportError:

  print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
