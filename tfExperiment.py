import Resnet
import getParams
import tensorflow as tf
import cv2
import numpy as np


files = ["/home/manish/projects/objectiveTF/Data/img10.jpeg","/home/manish/projects/objectiveTF/Data/img12.jpeg"]
images = np.zeros((len(files),224,224,3))
for i,f in enumerate(files):
    img = cv2.imread(f)
    img = cv2.resize(img,(224,224),interpolation=cv2.INTER_LINEAR)
    images[i, :, :, :] = img


with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False, name='global_step')
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels = tf.placeholder(tf.int32, [2])
    network = Resnet.resnet_v1(18,1000)
    initWeights = getParams.getWeights()
    x = tf.Variable(0)
    condition = tf.greater(x,1)
    preds = network(images,condition,initWeights)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        output = sess.run([logits],
                          feed_dict={x: images})