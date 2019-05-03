import math
import time
import tensorflow as tf
import numpy as np

#这里总共测试100个batch的数据。
batch_size=32
num_batches=100
#定义一个用来显示网络每一层结构的函数print_activations，展示每一个卷积层或池化层输出的tensor尺寸。
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

#设计AlexNet网络结构。
#设定inference函数，用于接受images作为输入，返回最后一层pool5（第5个池化层）及parameters（AlexnNet中所有需要训练的模型参数）
#该函数包括多个卷积层和池化层。
def inference(images,keep_prob):
    parameters = []
    # 第1个卷积层
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]
 
  # 添加LRN层和最大池化层
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print_activations(pool1)
 
  # 设计第2个卷积层
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)
 
  # 对第2个卷积层的输出进行处理，同样也是先做LRN处理再做最大化池处理。
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    print_activations(pool2)
 
  # 设计第3个卷积层
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],dtype=tf.float32,stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)
 
  # 设计第4个卷积层
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],dtype=tf.float32,stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)
 
  # 设计第5个卷积层
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)
 
  # 最大池化层
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    print_activations(pool5)
    
    #全连接1
    shape = int(np.prod(pool5.get_shape()[1:]))
    pool5_flat = tf.reshape(pool5, [-1, shape])

    with tf.name_scope('fc1') as scope:
        fc1w = tf.Variable(tf.truncated_normal([shape, 4096], dtype=tf.float32,stddev = 1e-1),name = 'weight')
        fc1b = tf.Variable(tf.constant(0.1, shape=[4096], dtype=tf.float32), name='b')
        mat = tf.matmul(pool5_flat, fc1w)
        fc1l = tf.nn.bias_add(mat,fc1b)
        fc2 = tf.nn.relu(fc1l)
        parameters += [fc1w, fc1b]

    print_activations(fc1l)
    #全连接2
    with tf.name_scope('fc2') as scope:
        fc2w = tf.get_variable(scope+'w',shape=[4096, 4096], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        fc2b = tf.Variable(tf.constant(0.1, shape=[4096], dtype=tf.float32), name='b')
        mat = tf.matmul(fc2,fc2w)
        fc2l = tf.nn.bias_add(mat,fc2b)
        fc3 = tf.nn.relu(fc2l)
        parameters += [fc2w, fc2b]

    print_activations(fc2l)
    fc6_drop = tf.nn.dropout(fc3, keep_prob)
    
    #全连接3
    with tf.name_scope('fc3') as scope:
        fc3w = tf.get_variable(scope+'w',shape=[4096, 1000], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        fc3b = tf.Variable(tf.constant(0.1, shape=[1000], dtype=tf.float32), name='b')
        mat = tf.matmul(fc3,fc3w)
        fc3l = tf.nn.bias_add(mat,fc3b)
        fc4 = tf.nn.relu(fc3l)
        parameters += [fc3w, fc3b]

    print_activations(fc3l)
    softmax = tf.nn.softmax(fc4)
    #print(softmax.get_shape())
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc4, parameters
 
#主函数
def run_benchmark():
 
    with tf.Graph().as_default():
        image_size = 227
        images = tf.Variable(tf.random_normal([batch_size,image_size,image_size, 3],dtype=tf.float32,stddev=1e-1))
        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc4, parameters = inference(images,keep_prob)
        #init = tf.global_variables_initializer()
        #sess = tf.Session()
        #Writer = tf.summary.FileWriter("./Net",sess.graph)
        #Writer.close()
        #sess.run(init)
    return

if __name__ == "__main__":
    run_benchmark()
