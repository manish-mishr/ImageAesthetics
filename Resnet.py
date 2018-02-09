

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.training import moving_averages

_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_DECAY = 0.997



def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())


def _batch_norm_layer(inputs, init, name, is_training):
  """Performs a batch normalization followed by a ReLU."""


  with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
      shape = inputs.get_shape().as_list()

      gamma = tf.get_variable('gamma',  initializer=tf.constant(init['weight']), trainable=True)
      beta = tf.get_variable('beta', initializer=tf.constant(init['bias']), trainable=True)
      moving_avg = tf.get_variable('moving_avg',initializer=tf.constant(init['running_mean']),trainable=False )
      moving_var = tf.get_variable('moving_var',  initializer=tf.constant(init['running_var']), trainable=False)


      avg, var = tf.nn.moments(inputs, range(len(shape)-1))
      update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, _BATCH_NORM_DECAY)
      update_moving_var = moving_averages.assign_moving_average(moving_var, var, _BATCH_NORM_DECAY)
      control_inputs = [update_moving_avg, update_moving_var]

      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_avg)
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_var)

      avg, var = tf.cond(is_training,
                         lambda:(avg, var),
                         lambda: (moving_avg, moving_var))

      output = tf.nn.batch_normalization(inputs,avg, var, offset=beta, scale=gamma, variance_epsilon=_BATCH_NORM_EPSILON)
  return output


def _batch_norm(inputs, init, name, is_training):
    return _batch_norm_layer(inputs,init, name, is_training , reuse=None)
    # return tf.cond(
    #     is_training,
    #     lambda: batch_norm_layer(inputs,init, name, is_training = True, reuse=True),
    #     lambda: batch_norm_layer(inputs, init, name,is_training = False, reuse=True)
    # )



def _conv2d(inputs, strides, init, pad='SAME', name='conv'):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        initTensor = tf.constant(init['weight'])
        kernel = tf.get_variable('kernel', initializer=initTensor)
    return tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], pad)



def _relu(inputs, leakness=0.0, name=None):
    name = 'relu' if name is None else 'lrelu'
    if leakness > 0.0:
        return tf.maximum(inputs, inputs*leakness, name=name)
    else:
        return tf.nn.relu(inputs, name=name)

def final_layer(inputs,init, name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
        inputs = tf.reshape(inputs,[-1,512])
        weights = tf.get_variable('Weights', initializer=tf.constant(init['wt']))
        bias = tf.get_variable('bias', initializer=tf.constant(init['bias']))
        inputs = tf.nn.bias_add(tf.matmul(inputs,weights),bias)
        inputs = tf.nn.softmax(inputs)
    return inputs


def feature_weights(inputs, shape, init, name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
        if init is None:
            weights = tf.get_variable('Weights', shape=shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer)
            bias = tf.get_variable('bias', shape=shape[-1], dtype=tf.float32, initializer=tf.truncated_normal_initializer)
        else:
            weights = tf.get_variable('Weights', initializer=tf.constant(init['wt']))
            bias = tf.get_variable('bias', initializer=tf.constant(init['bias']))

        inputs = tf.nn.bias_add(tf.matmul(inputs, weights), bias)
        inputs = tf.nn.softmax(inputs)
    return inputs


def building_block(inputs,  is_training, init, projection_shortcut, strides,
                   name):
  """Standard building block for residual networks with BN before convolutions.
    The output tensor of the block.
  """
  with tf.variable_scope(name) as scope:
      shortcut = inputs

      if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs,init['downsample'])

      inputs = _conv2d(
          inputs=inputs, init=init['conv1'], strides=strides,
          name='conv1')

      inputs = _batch_norm_layer(inputs, init['bn1'], name='bn1',is_training=is_training)
      inputs = _relu(inputs)

      inputs = _conv2d(
          inputs=inputs,  init=init['conv2'], strides=1,
          name='conv2')
      inputs = _batch_norm_layer(inputs, init['bn2'], name='bn2', is_training=is_training)

      inputs += shortcut
      inputs = _relu(inputs)

  return inputs




def block_layer(inputs, block_fn, init, blocks, strides, is_training,
                name):
  """Creates one layer of blocks for the ResNet model.
  Returns:
    The output tensor of the block layer.
  """
  with tf.variable_scope(name) as scope:
      def projection_shortcut(inputs, init):
        inputs =  _conv2d(
            inputs=inputs, init=init['0'], strides=strides)
        return _batch_norm_layer(inputs,  init=init['1'], name='bn', is_training=is_training)

      # Only the first block per block_layer uses projection_shortcut and strides
      projection_fn = projection_shortcut if strides > 1 else None

      inputs = block_fn(inputs,  is_training, init['0'], projection_fn, strides, 'block0')

      for ind in range(1, blocks):
        inputs = block_fn(inputs,  is_training, init[str(ind)], None, 1, 'block'+str(ind))

  return inputs

def objective_resnet_model(block_fn, layers):
  """Generator for ImageNet ResNet v2 models.
  """
  def model(inputs, is_training, init):
    """Constructs the ResNet model given the inputs."""
    with tf.variable_scope('initial_conv') as scope:
        inputs = _conv2d(
            inputs=inputs, strides=2, init=init['conv1']
        )
        inputs = _batch_norm_layer(inputs,init=init['bn1'], name='bn', is_training=is_training)
        inputs = _relu(inputs)

        inputs = tf.nn.max_pool(
            inputs, [1,3,3,1], [1, 2, 2, 1], padding='SAME')


    inputs = block_layer(
        inputs=inputs,block_fn=block_fn, init=init['layer1'], blocks=layers[0],
        strides=1, is_training=is_training, name='block_layer1'
    )
    inputs = block_layer(
        inputs=inputs, block_fn=block_fn, init=init['layer2'], blocks=layers[1],
        strides=2, is_training=is_training, name='block_layer2',
    )
    inputs = block_layer(
        inputs=inputs, block_fn=block_fn, init=init['layer3'],  blocks=layers[2],
        strides=2, is_training=is_training, name='block_layer3',
    )
    inputs = block_layer(
        inputs=inputs, block_fn=block_fn, init=init['layer4'], blocks=layers[3],
        strides=2, is_training=is_training, name='block_layer4',
    )


    inputs = tf.nn.avg_pool(
        value=inputs, ksize=[1,7,7,1], strides=[1,1,1,1], padding='VALID',name='final_avg_pool'
    )

    inputs = tf.reshape(inputs, [-1, 512])
    # newInput = tf.identity(inputs,name='newInp')
    # newOutput = feature_weights(newInput,[512,512],name='newOut')

    balancingInp = tf.identity(inputs,name='balancingInp')
    balancingOut = feature_weights(balancingInp, [512, 256], None, name='balancingOut')
    balanceScore = feature_weights(balancingOut,[256, 1], init['BalancingElement'], name='balanceScore')

    colorHarmonyInp = tf.identity(inputs, name='colorHarmonyInp')
    colorHarmonyOut = feature_weights(colorHarmonyInp, [512, 256], None, name='colorHarmonyOut')
    colorHarmonyscore = feature_weights(colorHarmonyOut, [256, 1], init['ColorHarmony'], name='colorHarmonyScore')

    contentInp = tf.identity(inputs, name='contentInp')
    contentOut = feature_weights(contentInp, [512, 256], None, name='contentOut')
    contentscore = feature_weights(contentOut, [256, 1], init['Content'], name='contentScore')

    DoFInp = tf.identity(inputs, name='DoFInp')
    DoFOut = feature_weights(DoFInp, [512, 256], None, name='DoFOut')
    DoFscore = feature_weights(DoFOut, [256, 1], init['DoF'], name='DoFScore')

    lightInp = tf.identity(inputs, name='lightInp')
    lightOut = feature_weights(lightInp, [512, 256], None, name='lightOut')
    lightscore = feature_weights(lightOut, [256, 1], init['Light'], name='lightScore')

    motionBlurInp = tf.identity(inputs, name='motionBlurInp')
    motionBlurOut = feature_weights(motionBlurInp, [512, 256], None, name='motionBlurOut')
    motionBlurscore = feature_weights(motionBlurOut, [256, 1], init['MotionBlur'], name='motionBlurScore')

    objectInp = tf.identity(inputs, name='objectInp')
    objectOut = feature_weights(objectInp, [512, 256], None, name='objectOut')
    objectscore = feature_weights(objectOut, [256, 1], init['Object'], name='objectScore')

    repetitionInp = tf.identity(inputs, name='repetitionInp')
    repetitionOut = feature_weights(repetitionInp, [512, 256], None, name='repetitionOut')
    repetitionscore = feature_weights(repetitionOut, [256, 1], init['Repetition'], name='repetitionScore')

    ruleOfThirdInp = tf.identity(inputs, name='ruleOfThirdInp')
    ruleOfThirdOut = feature_weights(ruleOfThirdInp, [512, 256], None, name='ruleOfThirdOut')
    ruleOfThirdscore = feature_weights(ruleOfThirdOut, [256, 1], init['RuleOfThirds'], name='ruleOfThirdScore')

    symmetryInp = tf.identity(inputs, name='symmetryInp')
    symmetryOut = feature_weights(symmetryInp, [512, 256], None, name='symmetryOut')
    symmetryscore = feature_weights(symmetryOut, [256, 1], init['Symmetry'], name='symmetryScore')

    vividColorInp = tf.identity(inputs, name='vividColorInp')
    vividColorOut = feature_weights(vividColorInp, [512, 256], None, name='vividColorOut')
    vividColorscore = feature_weights(vividColorOut, [256, 1], init['VividColor'], name='vividColorScore')

    objectiveScore = tf.concat([
        balanceScore,
        colorHarmonyscore,
        contentscore,
        DoFscore,
        lightscore,
        motionBlurscore,
        objectscore,
        repetitionscore,
        ruleOfThirdscore,
        symmetryscore,
        vividColorscore,
        ],axis=1)
    print_activations(objectiveScore)
    # inputs = final_layer(inputs,init['fc'], name='fc1')

    return objectiveScore

  return model


def resnet_v1(resnet_size):
  """Returns the ResNet model for a given size and number of output classes."""
  model_params = {
      18: {'block': building_block, 'layers': [2, 2, 2, 2]},
      34: {'block': building_block, 'layers': [3, 4, 6, 3]},
  }

  if resnet_size not in model_params:
    raise ValueError('Not a valid resnet_size:', resnet_size)

  params = model_params[resnet_size]
  return objective_resnet_model(
      params['block'], params['layers'])

