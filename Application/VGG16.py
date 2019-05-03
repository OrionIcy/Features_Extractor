from datetime import datetime
import numpy as np
import math
import time
import tensorflow as tf
#
def VGGNet(images, keep_prob):
    p = []

    #conv1_1 = convLayer(x, name='conv1_1', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    with tf.name_scope('Conv1_1') as scope:
        kernel = tf.get_variable(scope+'w',shape = [3,3,3,64],dtype = tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        bias_init = tf.constant(0.0, shape=[64], dtype=tf.float32)
        biases = tf.Variable(bias_init, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        conv1_1 = activation

    #conv1_2 = convLayer(conv1_1, name='conv1_2', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    with tf.name_scope('Conv1_2') as scope:
        kernel = tf.get_variable(scope+'w',shape=[3, 3, 64, 64], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        bias_init = tf.constant(0.0, shape=[64], dtype=tf.float32)
        biases = tf.Variable(bias_init, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        conv1_2 = activation

    pool1 = tf.nn.max_pool(conv1_2,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')

    #conv2_1 = convLayer(pool1, name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    with tf.name_scope('Conv2_1') as scope:
        kernel = tf.get_variable(scope+'w',shape=[3, 3, 64, 128], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        bias_init = tf.constant(0.0, shape=[128], dtype=tf.float32)
        biases = tf.Variable(bias_init, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        conv2_1 = activation

    #conv2_2 = convLayer(conv2_1, name='conv2_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    with tf.name_scope('Conv2_2') as scope:
        kernel = tf.get_variable(scope+'w',shape=[3, 3, 128, 128], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        bias_init = tf.constant(0.0, shape=[128], dtype=tf.float32)
        biases = tf.Variable(bias_init, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        conv2_2 = activation

    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    #conv3_1 = convLayer(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    with tf.name_scope('Conv3_1') as scope:
        kernel = tf.get_variable(scope+'w',shape=[3, 3, 128, 256], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        bias_init = tf.constant(0.0, shape=[256], dtype=tf.float32)
        biases = tf.Variable(bias_init, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        conv3_1 = activation

    #conv3_2 = convLayer(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    with tf.name_scope('Conv3_2') as scope:
        kernel = tf.get_variable(scope+'w',shape=[3, 3, 256, 256], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        bias_init = tf.constant(0.0, shape=[256], dtype=tf.float32)
        biases = tf.Variable(bias_init, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        conv3_2 = activation

    #conv3_3 = convLayer(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    with tf.name_scope('Conv3_3') as scope:
        kernel = tf.get_variable(scope+'w',shape=[3, 3, 256, 256], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        bias_init = tf.constant(0.0, shape=[256], dtype=tf.float32)
        biases = tf.Variable(bias_init, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        conv3_3 = activation

    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    #conv4_1 = convLayer(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    with tf.name_scope('Conv4_1') as scope:
        kernel = tf.get_variable(scope+'w',shape=[3, 3, 256, 512], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        bias_init = tf.constant(0.0, shape=[512], dtype=tf.float32)
        biases = tf.Variable(bias_init, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        conv4_1 = activation

    #conv4_2 = convLayer(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    with tf.name_scope('Conv4_2') as scope:
        kernel = tf.get_variable(scope+'w',shape=[3, 3, 512, 512], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        bias_init = tf.constant(0.0, shape=[512], dtype=tf.float32)
        biases = tf.Variable(bias_init, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        conv4_2 = activation

    #conv4_3 = convLayer(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    with tf.name_scope('Conv4_3') as scope:
        kernel = tf.get_variable(scope+'w',shape=[3, 3, 512, 512], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        bias_init = tf.constant(0.0, shape=[512], dtype=tf.float32)
        biases = tf.Variable(bias_init, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        conv4_3 = activation

    pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    #conv5_1 = convLayer(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    with tf.name_scope('Conv5_1') as scope:
        kernel = tf.get_variable(scope+'w',shape=[3, 3, 512, 512], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        bias_init = tf.constant(0.0, shape=[512], dtype=tf.float32)
        biases = tf.Variable(bias_init, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        conv5_1 = activation

    #conv5_2 = convLayer(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    with tf.name_scope('Conv5_2') as scope:
        kernel = tf.get_variable(scope+'w',shape=[3, 3, 512, 512], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        bias_init = tf.constant(0.0, shape=[512], dtype=tf.float32)
        biases = tf.Variable(bias_init, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        conv5_2 = activation

    #conv5_3 = convLayer(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    with tf.name_scope('Conv5_3') as scope:
        kernel = tf.get_variable(scope+'w',shape=[3, 3, 512, 512], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        bias_init = tf.constant(0.0, shape=[512], dtype=tf.float32)
        biases = tf.Variable(bias_init, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        conv5_3 = activation

    pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    print(pool5.get_shape().as_list())
    
    
    
    shape = int(np.prod(pool5.get_shape()[1:]))
    pool5_flat = tf.reshape(pool5, [-1, shape])

    with tf.name_scope('FC1') as scope:
        fc1w = tf.Variable(tf.truncated_normal([shape, 4096], dtype=tf.float32,stddev = 1e-1),name = 'weight')
        fc1b = tf.Variable(tf.constant(0.1, shape=[4096], dtype=tf.float32), name='b')
        mat = tf.matmul(pool5_flat, fc1w)
        fc1l = tf.nn.bias_add(mat,fc1b)
        fc2 = tf.nn.relu(fc1l)
        p += [fc1w, fc1b]

    #fc6 = fcLayer(fc2, name='fc6', n_out=4096, p=p)
    with tf.name_scope('FC2') as scope:
        fc2w = tf.get_variable(scope+'w',shape=[4096, 4096], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        fc2b = tf.Variable(tf.constant(0.1, shape=[4096], dtype=tf.float32), name='b')
        mat = tf.matmul(fc2,fc2w)
        fc2l = tf.nn.bias_add(mat,fc2b)
        fc3 = tf.nn.relu(fc2l)
        p += [fc2w, fc2b]

    fc6_drop = tf.nn.dropout(fc3, keep_prob)
    
    #fc7_drop = fcLayer(fc6_drop, name='fc7', n_out=4096, p=p)
    with tf.name_scope('FC3') as scope:
        fc3w = tf.get_variable(scope+'w',shape=[4096, 1000], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        fc3b = tf.Variable(tf.constant(0.1, shape=[1000], dtype=tf.float32), name='b')
        mat = tf.matmul(fc3,fc3w)
        fc3l = tf.nn.bias_add(mat,fc3b)
        fc4 = tf.nn.relu(fc3l)
        p += [fc3w, fc3b]

    softmax = tf.nn.softmax(fc4)
    print(softmax.get_shape())
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc4, p



#
def run_benchmark():
    with tf.Graph().as_default():
        batch_size = 32
        image_size = 224
        images = tf.Variable(tf.random_normal([32,224,224,3],dtype=tf.float32,stddev=1e-1))

        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, p =VGGNet(images, keep_prob)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        Writer = tf.summary.FileWriter("./Net",sess.graph)
        Writer.close()
        sess.run(init)
        #time_tensorflow_run(sess, predictions, {keep_prob:1.0}, 'Froward')
        #objective = tf.nn.l2_loss(fc8)
        #grad = tf.gradients(objective, p)
        #time_tensorflow_run(sess, grad, {keep_prob:0.5}, 'Forward-backward')


if __name__ == '__main__':
    run_benchmark()
