import Resnet
import getParams
import tensorflow as tf
from PIL import Image
import numpy as np


files = ["/home/manish/projects/objectiveTF/Data/img10.jpeg","/home/manish/projects/objectiveTF/Data/img12.jpeg"]
filesImg = np.zeros((len(files),224,224,3))
for i,f in enumerate(files):
    img = Image.open(f)
    img = img.resize((224,224),Image.ANTIALIAS)
    img = np.array(img, np.float32)
    filesImg[i, :, :, :] = img


with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False, name='global_step')
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels = tf.placeholder(tf.int32, [2])
    network = Resnet.resnet_v1(18)
    initWeights = getParams.getWeights()
    x = tf.Variable(0)
    condition = tf.greater(x,1)
    preds = network(images,condition,initWeights)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        output = sess.run([preds],feed_dict={images: filesImg})