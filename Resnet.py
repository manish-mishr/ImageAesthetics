

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.training import moving_averages

_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_DECAY = 0.997






def batch_norm_layer(inputs, init, name, is_training, reuse=None):
  """Performs a batch normalization followed by a ReLU."""


  with tf.variable_scope(name,reuse=reuse):
      shape = inputs.get_shape().as_list()

      gamma = tf.get_variable('gamma',  initializer=tf.constant(init['weight']), trainable=True)
      beta = tf.get_variable('beta', initializer=tf.constant(init['bias']), trainable=True)
      moving_avg = tf.get_variable('moving_avg',initializer=tf.constant(init['running_mean']),trainable=False )
      moving_var = tf.get_variable('moving_var',  initializer=tf.constant(init['running_var']), trainable=False)


      avg, var = tf.nn.moments(inputs, range(len(shape)-1))
      update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, _BATCH_NORM_DECAY)
      update_moving_var = moving_averages.assign_moving_average(moving_var, var, _BATCH_NORM_DECAY)
      control_inputs = [update_moving_avg, update_moving_var]

      # avg, var, control_inputs = tf.cond(is_training,
      #                                    lambda:(avg, var, control_inputs),
      #                                    lambda: (moving_avg, moving_var, []))
      with tf.control_dependencies(control_inputs):
          output = tf.nn.batch_normalization(inputs,avg, var, offset=beta, scale=gamma, variance_epsilon=_BATCH_NORM_EPSILON)
  return output


def _batch_norm(inputs, is_training, init, name='bn'):
    return batch_norm_layer(inputs,init, name, is_training , reuse=None)
    # return tf.cond(
    #     is_training,
    #     lambda: batch_norm_layer(inputs,init, name, is_training = True, reuse=True),
    #     lambda: batch_norm_layer(inputs, init, name,is_training = False, reuse=True)
    # )



def _conv2d(inputs, out_channel, kernel_size, strides, init, pad='SAME', name='conv'):

    in_channel = (inputs.get_shape().as_list())[-1]
    with tf.variable_scope(name):
        initTensor = tf.constant(init['weight'])
        kernel = tf.get_variable('kernel', initializer=initTensor)
    return tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], pad)



def _relu(inputs, leakness=0.0, name=None):
    name = 'relu' if name is None else 'lrelu'
    if leakness > 0.0:
        return tf.maximum(inputs, inputs*leakness, name=name)
    else:
        return tf.nn.relu(inputs, name=name)

def final_layer(inputs,init, num_classes, name):
    with tf.variable_scope(name) as scope:
        inputs = tf.reshape(inputs,[-1,512])
        weights = tf.get_variable('Weights', initializer=tf.constant(init['weight']))
        bias = tf.get_variable('bias', initializer=tf.constant(init['bias']))
        inputs = tf.nn.bias_add(tf.matmul(inputs,weights),bias)
        inputs = tf.nn.softmax(inputs)
    return inputs


def building_block(inputs, out_channel, is_training, init, projection_shortcut, strides,
                   name):
  """Standard building block for residual networks with BN before convolutions.
    The output tensor of the block.
  """
  with tf.variable_scope(name) as scope:
      shortcut = inputs

      if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs,init['downsample'])

      inputs = _conv2d(
          inputs=inputs,out_channel=out_channel,kernel_size=3, init=init['conv1'], strides=strides,
          name='conv1')

      inputs = _batch_norm(inputs, is_training, init['bn1'], name='bn1')
      inputs = _relu(inputs)

      inputs = _conv2d(
          inputs=inputs,  out_channel=out_channel,kernel_size=3, init=init['conv2'], strides=1,
          name='conv2')
      inputs = _batch_norm(inputs, is_training, init['bn2'],name='bn2')

      inputs += shortcut
      inputs = _relu(inputs)

  return inputs




def block_layer(inputs, out_channel,block_fn, init, blocks, strides, is_training,
                name):
  """Creates one layer of blocks for the ResNet model.
  Returns:
    The output tensor of the block layer.
  """
  with tf.variable_scope(name) as scope:
      def projection_shortcut(inputs, init):
        inputs =  _conv2d(
            inputs=inputs,  out_channel=out_channel, kernel_size=1,init=init['0'], strides=strides)
        return _batch_norm(inputs, is_training, init=init['1'])

      # Only the first block per block_layer uses projection_shortcut and strides
      projection_fn = projection_shortcut if strides > 1 else None

      inputs = block_fn(inputs, out_channel, is_training, init['0'], projection_fn, strides, 'block0')

      for ind in range(1, blocks):
        inputs = block_fn(inputs, out_channel, is_training, init[str(ind)], None, 1, 'block'+str(ind))

  return inputs

def objective_resnet_model(block_fn, layers,  num_classes):
  """Generator for ImageNet ResNet v2 models.
  """
  def model(inputs, is_training, init):
    """Constructs the ResNet model given the inputs."""
    with tf.variable_scope('initial_conv') as scope:
        inputs = _conv2d(
            inputs=inputs, out_channel=64, kernel_size=7, strides=2, init=init['conv1']
        )
        inputs = _batch_norm(inputs,is_training,init=init['bn1'])
        inputs = _relu(inputs)

        inputs = tf.nn.max_pool(
            inputs, [1,3,3,1], [1, 2, 2, 1], padding='SAME')


    inputs = block_layer(
        inputs=inputs, out_channel=64, block_fn=block_fn, init=init['layer1'], blocks=layers[0],
        strides=1, is_training=is_training, name='block_layer1'
    )
    inputs = block_layer(
        inputs=inputs, out_channel=128, block_fn=block_fn, init=init['layer2'], blocks=layers[1],
        strides=2, is_training=is_training, name='block_layer2',
    )
    inputs = block_layer(
        inputs=inputs, out_channel=256, block_fn=block_fn, init=init['layer3'],  blocks=layers[2],
        strides=2, is_training=is_training, name='block_layer3',
    )
    inputs = block_layer(
        inputs=inputs, out_channel=512, block_fn=block_fn, init=init['layer4'], blocks=layers[3],
        strides=2, is_training=is_training, name='block_layer4',
    )


    inputs = tf.nn.avg_pool(
        value=inputs, ksize=[1,7,7,1], strides=[1,1,1,1], padding='VALID',name='final_avg_pool'
    )

    inputs = final_layer(inputs,init['fc'], num_classes, name='fc')

    return inputs

  return model


def resnet_v1(resnet_size, num_classes):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': building_block, 'layers': [2, 2, 2, 2]},
      34: {'block': building_block, 'layers': [3, 4, 6, 3]},
  }

  if resnet_size not in model_params:
    raise ValueError('Not a valid resnet_size:', resnet_size)

  params = model_params[resnet_size]
  return objective_resnet_model(
      params['block'], params['layers'], num_classes)

