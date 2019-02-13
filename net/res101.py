import tensorflow as tf

slim = tf.contrib.slim


def build_res101(inputs):
    net = slim.conv2d(inputs, 64, 7, stride=2, padding='same', scope='conv1')

    net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='pool1')

    with tf.variable_scope('block1'):
        for x in range(3):
            with tf.variable_scope('unit_' + str(x + 1)):
                net = bottleneck(inputs=net, depth=64 * 4, depth_bottleneck=64, stride=1)
    C2 = net

    with tf.variable_scope('block2'):
        for x in range(4):
            stride = 2 if x == 0 else 1
            with tf.variable_scope('unit_' + str(x + 1)):
                net = bottleneck(inputs=net, depth=128 * 4, depth_bottleneck=128, stride=stride)
    C3 = net
    with tf.variable_scope('block3'):
        for x in range(23):
            stride = 2 if x == 0 else 1
            with tf.variable_scope('unit_' + str(x + 1)):
                net = bottleneck(inputs=net, depth=256 * 4, depth_bottleneck=256, stride=stride)
    C4 = net
    with tf.variable_scope('block4'):
        for x in range(3):
            stride = 2 if x == 0 else 1
            with tf.variable_scope('unit_' + str(x + 1)):
                net = bottleneck(inputs=net, depth=512 * 4, depth_bottleneck=512, stride=stride)
    C5 = net
    return C2, C3, C4, C5


def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               scope=None):
    """Bottleneck residual unit variant with BN after convolutions.
    This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
    its definition. Note that we use here the bottleneck variant which has an
    extra bottleneck layer.
    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.
    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.

      scope: Optional variable_scope.

    """
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]):
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(
                inputs,
                depth, [1, 1],
                stride=stride,
                activation_fn=None,
                scope='shortcut')

        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv2d_same(inputs=residual, num_outputs=depth_bottleneck, kernel_size=3, stride=stride,
                               scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')

        output = tf.nn.relu(shortcut + residual)

        return output


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    """Strided 2-D convolution with 'SAME' padding.
    When stride > 1, then we do explicit zero-padding, followed by conv2d with
    'VALID' padding.
    Note that
       net = conv2d_same(inputs, num_outputs, 3, stride=stride)
    is equivalent to
       net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
       net = subsample(net, factor=stride)
    whereas
       net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')
    is different when the input's height or width is even, which is why we add the
    current function. For more details, see ResnetUtilsTest.testConv2DSameEven().
    Args:
      inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
      num_outputs: An integer, the number of output filters.
      kernel_size: An int with the kernel_size of the filters.
      stride: An integer, the output stride.
      rate: An integer, rate for atrous convolution.
      scope: Scope.
    Returns:
      output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                           padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           rate=rate, padding='VALID', scope=scope)


def subsample(inputs, factor, scope=None):
    """Subsamples the input along the spatial dimensions.
    Args:
      inputs: A `Tensor` of size [batch, height_in, width_in, channels].
      factor: The subsampling factor.
      scope: Optional variable_scope.
    Returns:
      output: A `Tensor` of size [batch, height_out, width_out, channels] with the
        input, either intact (if factor == 1) or subsampled (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

