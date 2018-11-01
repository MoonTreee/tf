import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf

url = 'http://mattmahoney.net/dc/'
vocabulary_size = 50000
data_index = 0


def download_data(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)

    statinfo = os.stat(filename)

    if statinfo.st_size == expected_bytes:
        print("Found and verified ", filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            "Fail to verify " + filename
        )
    return filename


def read__data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        # 对字典里面的词进行编号
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0

    # 转化成index的形式，如果不在字典中就转换成0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


"""
@:param batch 
@:param num_skips 对每个单词生成的样本数
@:param skip_windows 单词最远可到达的距离
"""


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % skip_window == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # span是每个单词构建样本时候所需要的单词的数量
    span = 2 * skip_window + 1

    # 构建一个队列buffer，存放的是目标单词和其他所有的相关词
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips):
        target = skip_window
        target_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in target_to_avoid:
                target = random.randint(0, span - 1)
            target_to_avoid.append(target)
            # 目标词
            batch[i * num_skips + j] = buffer[skip_window]
            # 标签（相关词）
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels



batch_size = 128
embedding_size = 128
skip_window = 1
num__skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64
graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device("/cpu:0"):
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
        )
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        nce_weight = tf.Variable(
            tf.truncated_normal([vocabulary_size], stddev=1.0 / math.sqrt(embedding_size))
        )

        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                         biases=nce_biases),
                          labels=train_labels,
                          inputs=embed,
                          num_sampled=num_sampled,
                          num_classes=vocabulary_size))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embedding = embeddings/norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embedding, transpose_b=True
    )
    init = tf.global_variables_initializer()


if __name__ == "__main__":
    words = read__data("text8.zip")
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]], "->", labels[i, 0], reverse_dictionary[labels[i, 0]])




