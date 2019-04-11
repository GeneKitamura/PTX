import numpy as np
import tensorflow as tf

from utils import safe_sigmoid, exp_norm_softmax

UNIT_STRIDE = [1, 1, 1, 1]
DOUBLE_STRIDE = [1,2,2,1]
SAME_STR = 'SAME'
VALID_STR = 'VALID'

def _init_weight(name, shape, alpha, initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=True):
    var = tf.get_variable(name, shape, np.float32, initializer=initializer, trainable=trainable)
    if alpha is not None:
        l2_loss = alpha * tf.nn.l2_loss(var)
        tf.add_to_collection('l2_loss', l2_loss)
    return var

def _init_bias(name, out_channels, q_val=0.1, trainable=True):
    var = tf.get_variable(name, [out_channels], initializer=tf.constant_initializer(q_val), dtype=np.float32, trainable=trainable)
    return var


def fractionally_strided_conv(l, kernel_sz, out_dims, strides, padding, alpha=None):
    batch_sz = tf.shape(l)[0] #tf op
    c = l.shape[3].value #tensor values.  Need to actually get value rather than tensor Dimensions for the tfn..n.conv2d_tranpose part
    w = _init_weight('W', [kernel_sz, kernel_sz, c, c], alpha)
    l = tf.nn.conv2d_transpose(l, w, [batch_sz, out_dims, out_dims, c], [1, strides, strides, 1], padding)
    l = tf.reshape(l, [batch_sz, out_dims, out_dims, c]) #otherwise shape is unknown
    return l

def simple_conv(l, kernel_sz, out_channel, stride, padding, alpha):
    weight = _init_weight('W', [kernel_sz, kernel_sz, l.shape[3], out_channel], alpha=alpha)
    l = tf.nn.conv2d(l, weight, [1, stride, stride, 1], padding)
    return l

def conv_bn_relu(l, kernel_sz, out_channel, stride, padding, training, alpha=None, preact=False, trainable=True):

    if not preact:
        weight = _init_weight('W', [kernel_sz, kernel_sz, l.shape[3], out_channel], alpha=alpha, trainable=trainable)
        l = tf.nn.conv2d(l, weight, [1, stride, stride, 1], padding)
        l = tf.layers.batch_normalization(l, training=training, name='bn', fused=True, trainable=trainable)
        l = tf.nn.relu(l, name='relu')

    else:
        l = tf.layers.batch_normalization(l, training=training, name='bn', fused=True, trainable=trainable)
        l = tf.nn.relu(l, name='relu')
        weight = _init_weight('W', [kernel_sz, kernel_sz, l.shape[3], out_channel], alpha=alpha, trainable=trainable)
        l = tf.nn.conv2d(l, weight, [1, stride, stride, 1], padding)

    return l

def projection(data_in, out_channel, training, down_sample=True, alpha=None, trainable=True):
    kernel = _init_weight('W', [1, 1, data_in.shape[3], out_channel], alpha, trainable=trainable)
    conv = tf.nn.conv2d(data_in, kernel, strides=DOUBLE_STRIDE if down_sample else UNIT_STRIDE, padding='SAME')
    out = tf.layers.batch_normalization(conv, axis=-1, training=training, name='bn', fused=True, trainable=trainable)
    return out

def Resnet(x, is_training, n_classes, weight_decay, include_top=False, preact=False, depth=101, train_base_resnet=True):
    # IMPORTANT POINTS:
    # For pre-activation Residual Units, make sure to do RELU BEFORE the first path split/addition and AFTER the last element-wise addition.
    assert (depth == 50 or depth == 101), "Depth needs to be either 50 or 101"

    long_repeat = 23 if depth == 101 else 6
    layers = [3, 4, long_repeat, 3]

    def _add_units(inp_1, inp_2):
        return tf.add_n([inp_1, inp_2], name='add_units')

    # Units strides with SAME padding results in NO change in feature_map sizes
    def _residual_unit(data_in, f_1x1_1, f_3x3, f_1x1_2, downsampling):
        with tf.variable_scope('conv1'):
            conv_1x1_1 = conv_bn_relu(data_in, 1, f_1x1_1, 1, SAME_STR, is_training, preact=preact, trainable=train_base_resnet)
        selected_stride = 1 if not downsampling else 2

        with tf.variable_scope('conv2'):
            conv_3x3 = conv_bn_relu(conv_1x1_1, 3, f_3x3, selected_stride, SAME_STR, is_training, preact=preact, trainable=train_base_resnet)

        with tf.variable_scope('conv3'):
            conv_1x1_2 = conv_bn_relu(conv_3x3, 1, f_1x1_2, 1, SAME_STR, is_training, preact=preact, trainable=train_base_resnet)

        return conv_1x1_2

    #equation for calculating shape-out is o = [i + 2*pad - kernel]/strides + 1
    with tf.variable_scope('conv0'):
        conv1_out = conv_bn_relu(x, 7, 64, 2, VALID_STR, is_training, preact=False, trainable=train_base_resnet)
        if preact:
            conv1_out = tf.nn.relu(conv1_out)


    max_pool_1 = tf.nn.max_pool(conv1_out, ksize=[1,3,3,1], strides=DOUBLE_STRIDE, name='max_pool_1', padding=SAME_STR)

    with tf.variable_scope('group0/block0/convshortcut'):
        proj_pool = projection(max_pool_1, 256, training=is_training, down_sample=False, trainable=train_base_resnet)

    with tf.variable_scope('group0/block0'):
        conv2_1 = _residual_unit(max_pool_1, 64, 64, 256, downsampling=False)
        prev_output = _add_units(proj_pool, conv2_1)

    #First multi-residual unit layer
    for i in range(layers[0]-1):
        with tf.variable_scope('group0/block{}'.format(i+1)):
            out_now = _residual_unit(prev_output, 64, 64, 256, downsampling=False)
            prev_output = _add_units(out_now, prev_output)

    # First layer of conv3 with downsampling
    # First projection since feature_map sizes are now going to be different.
    with tf.variable_scope('group1/block0'):
        conv3_1 = _residual_unit(prev_output, 128, 128, 512, downsampling=True)

    with tf.variable_scope('group1/block0/convshortcut'):
        proj_2_3 = projection(prev_output, 512, training=is_training, trainable=train_base_resnet)
        prev_output = _add_units(proj_2_3, conv3_1)

    #Second multi-residual unit layer from conv3_2 on
    for i in range(layers[1]-1):
        with tf.variable_scope('group1/block{}'.format(i+1)):
            out_now = _residual_unit(prev_output, 128, 128, 512, downsampling=False)
            prev_output = _add_units(out_now, prev_output)

    #2nd projection level
    with tf.variable_scope('group2/block0'):
        conv4_1 = _residual_unit(prev_output, 256, 256, 1024, downsampling=True)

    with tf.variable_scope('group2/block0/convshortcut'):
        proj_3_4 = projection(prev_output, 1024, training=is_training, trainable=train_base_resnet)
        prev_output = _add_units(conv4_1, proj_3_4)

    # Third multi-residual unit layer
    for i in range(layers[2] -1):
        with tf.variable_scope('group2/block{}'.format(i+1)):
            out_now = _residual_unit(prev_output, 256, 256, 1024, downsampling=False)
            prev_output = _add_units(out_now, prev_output)

    #3rd projection level
    with tf.variable_scope('group3/block0'):
        conv5_1 = _residual_unit(prev_output, 512, 512, 2048, downsampling=True)

    with tf.variable_scope('group3/block0/convshortcut'):
        proj_4_5 = projection(prev_output, 2048, training=is_training, trainable=train_base_resnet)
        prev_output = _add_units(conv5_1, proj_4_5)

    # Fourth/last multi-residual unit layer
    for i in range(layers[3] -1):
        with tf.variable_scope('group3/block{}'.format(i+1)):
            out_now = _residual_unit(prev_output, 512, 512, 2048, downsampling=False)
            prev_output = _add_units(out_now, prev_output)

    if preact:
        prev_output = tf.nn.relu(prev_output, 'post_relu')

    if include_top:
        with tf.variable_scope('linear'):
            glob_pool = tf.nn.avg_pool(prev_output, ksize=[1, prev_output.shape[1], prev_output.shape[2], 1], strides=UNIT_STRIDE, padding=VALID_STR, name='glob_pool')
            N, H, W, F = glob_pool.shape
            flattened = tf.reshape(glob_pool, [-1, int(H*W*F)], name='flattened')
            N, D = flattened.shape
            FC_W = _init_weight('W', shape=[D, n_classes], alpha=weight_decay)
            FC_b = _init_bias('b', n_classes)
            mul = tf.matmul(flattened, FC_W)
            logits = tf.add(mul, FC_b, 'logits')
            prev_output = logits

    return prev_output


def recognition_tail(l, training, classes, alpha_tail, P, frac_dict, softmax=True, alpha_head=None):
    with tf.variable_scope('patch_slicing'):
        l = fractionally_strided_conv(l, frac_dict['k'], P, frac_dict['s'], frac_dict['pad'], alpha=alpha_head)
    with tf.variable_scope('patch_features'):
        l = conv_bn_relu(l, 3, 512, 1, 'SAME', training, alpha=alpha_head)
    with tf.variable_scope('patch_scores'):
        patch_logits = simple_conv(l, 1, classes, 1, 'SAME', alpha_tail)
    if softmax:
        # With exp-normalization
        with tf.name_scope('patch_probs'):
            patch_probs = exp_norm_softmax(patch_logits)
    else:
        with tf.name_scope('patch_probs'):
            patch_probs = safe_sigmoid(patch_logits)
        #patch_probs = tf.nn.sigmoid(patch_logits, name='patch_probs')
    return patch_probs, patch_logits