import numpy as np
import tensorflow as tf

from patch_model import Resnet, recognition_tail
from utils import selective_initializer


class Recognition_Network:
    def __init__(self, classes, box_gamma, alpha_tail, sess, graph, P, frac_dict=None, eps=1e-7, train_base_resnet=True, alpha_head=None):
        self.graph = graph
        self.sess = sess
        self.classes = classes
        self.box_gamma = box_gamma
        self.alpha_tail = alpha_tail # only relevant at patch logits layer (simple conv)
        self.P = P
        self.eps = eps
        self.train_base_resnet = train_base_resnet
        self.alpha_head = alpha_head

        # Remember that Resnet 101 with inp of 300 will result in output of 10
        if self.P == 20:
            self.frac_dict = {'k': 3, 's': 2, 'pad': 'SAME', 'low': 0.98}
        elif self.P == 12:
            self.frac_dict = {'k': 3, 's': 1, 'pad': 'VALID', 'low': 0.94}
        else:
            self.frac_dict = frac_dict

    def set_session(self, sess):
        self.sess = sess

    def set_graph(self, graph):
        self.graph=graph

    def set_params(self, data_tup_el):
        self.x = data_tup_el.images
        self.labels = data_tup_el.labels
        self.nu = data_tup_el.nus
        self.singles = data_tup_el.singles
        self.is_training = tf.placeholder_with_default(True, None, 'is_training')

    def _build_resnet(self, resnet_path):

        self.res_out = Resnet(self.x, self.is_training, self.classes, self.alpha_tail, train_base_resnet=self.train_base_resnet)
        self.resnet_saver = tf.train.Saver(max_to_keep=3)

        if resnet_path is not None:
            print("Loading Resnet model from :{}".format(resnet_path))
            self.resnet_saver.restore(self.sess, resnet_path)

    def _build_graph(self, patch_model_path):
        self.patch_probs, self.patch_logits = recognition_tail(self.res_out, self.is_training, self.classes, self.alpha_tail, self.P, self.frac_dict, alpha_head=self.alpha_head) # shape batch x H x W x K (batch x P x P x K)
        self.patch_model_saver = tf.train.Saver(max_to_keep=3)

        if patch_model_path is not None:
            print("Loading patch model from :{}".format(patch_model_path))
            self.patch_model_saver.restore(self.sess, patch_model_path)

        self.minus_probs = 1 - self.patch_probs

        #rescaling from [0, 1] to [low, 1] to counter underflow problem
        probs_low_rescale = self.frac_dict['low']
        probs_high_rescale = 1.0
        with tf.name_scope('rescaled_patch_probs'):
            self.pos_probs = (self.patch_probs - 0) * (probs_high_rescale - probs_low_rescale) / (1 - 0) + probs_low_rescale
            self.minus_probs = (self.minus_probs - 0) * (probs_high_rescale - probs_low_rescale) / (1 - 0) + probs_low_rescale
        # Patch probs [0.94, 1], but when multiplied for image-level, expands to [0, 1].

        #self.log = {}

        self.loose_prob = {}
        bound_prob = {}

        # Loss function
        # With bounding boxes
        for k in range(self.classes):
            mask = tf.equal(self.labels[..., k], 1)
            positives = tf.cast(mask, tf.bool)
            negatives = tf.cast(tf.logical_not(mask), tf.bool)
            # Mask not going to work because 0 times 0 results in zero
            # Filtering does not preserve shape
            ones = tf.ones_like(self.pos_probs[..., k])
            bound_prob[k] = tf.reduce_prod(tf.where(positives, self.pos_probs[..., k], ones), axis=[1, 2]) * \
                            tf.reduce_prod(tf.where(negatives, self.minus_probs[..., k], ones), axis=[1, 2])
            # bound_prob[k] shape is input/batch size
            #self.log['bound_{}'.format(k)] = bound_prob[k]

        # No bounding boxes
        # for each class, look at all the patches and get probability for that class
        for k in range(self.classes):
            self.loose_prob[k] = 1 - tf.reduce_prod(self.minus_probs[..., k], axis=[1, 2]) # shape is input/batch size
            #self.log['loose_{}'.format(k)] = self.loose_prob[k]

        # Total loss
        for k in range(self.classes):
            curr_label = tf.cast(self.labels[..., 0, 0, k], tf.float32) # just picking a label for unbound images, size is batch_sz.  Will be either 0 or 1 based on current k and self.labels.

            with tf.name_scope('class_{}_loss'.format(k)):
                c_log_bound = tf.reduce_sum(self.nu * tf.log(bound_prob[k] + self.eps))
                c_log_loose_positive = tf.reduce_sum((1 - self.nu) * curr_label * tf.log(self.loose_prob[k] + self.eps))
                c_log_loose_negative = tf.reduce_sum((1 - self.nu) * (1 - curr_label) * tf.log(1 - self.loose_prob[k] + self.eps))

                k_loss = - self.box_gamma * c_log_bound - c_log_loose_positive - c_log_loose_negative

            #self.log['log_bound_{}'.format(k)] = c_log_bound
            #self.log['log_loose_positive_{}'.format(k)] = c_log_loose_positive
            #self.log['log_loose_negative_{}'.format(k)] = c_log_loose_negative

            tf.add_to_collection('class_loss', k_loss)

        #sum all loss across classes
        class_loss = tf.reduce_sum(tf.get_collection('class_loss'), name='class_loss')
        l2_loss = tf.reduce_sum(tf.get_collection('l2_loss'), name='l2_loss')

        total_loss = tf.add(class_loss, l2_loss, 'total_loss')

        self.losses = {
            'class': class_loss,
            'l2': l2_loss,
            'total': total_loss
        }


    def build_optimizer(self, lr=0.001, adam_eps=0.1, var_init=True, resnet_path=None, patch_model_path=None, total_model_path=None):
        self._build_resnet(resnet_path)
        self._build_graph(patch_model_path)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=adam_eps)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.losses['total'], global_step=self.global_step)

        self.total_model_saver = tf.train.Saver(max_to_keep=3)
        if total_model_path is not None:
            print("Loading total model from :{}".format(total_model_path))
            self.total_model_saver.restore(self.sess, total_model_path)

        if var_init:
            selective_initializer(self.sess, self.graph)

    def predict(self, is_training=False):
        loose_probs, patch_probs, labels = self.sess.run([self.loose_prob, self.patch_probs, self.labels], feed_dict={self.is_training: is_training})

        return loose_probs, patch_probs, labels

    def score(self, is_training=False):
        # for binary classification only
        loose_probs, singles = self.sess.run([self.loose_prob, self.singles], feed_dict={self.is_training: is_training})

        return loose_probs, singles

    def write_graph(self):
        writer = tf.summary.FileWriter('./logs/graphs', self.graph)