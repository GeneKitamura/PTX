#!/usr/bin/env python
from check_warnings import suppress_warnings, check_list
suppress_warnings()
check_list()

import tensorflow as tf
import numpy as np
import time
import argparse

from load_models import Recognition_Network
from utils import eval_metrics, create_holders, whole_sample_jaccard
from load_Datas import TFR_Dataset, Memory_Dataset, Dataset_iterator
from process import process_test_boxes

# TRAIN_ROOT = './INCEP_train_P12/*.tfr'
# TEST_ROOT = './INCEP_test_P12/*.tfr'

TRAIN_ROOT = None
TEST_ROOT = None

# UP_TRAIN_PRECROP = './p12_cal_315_PTXnu0.npz'
# UP_TRAIN_300 = './p12_cal_300_PTXnu0.npz'
# UP_TEST = './p12_val_300_PTXnu0.npz'

UP_TRAIN_PRECROP = None
UP_TRAIN_300 = None
UP_TEST = './val_P20_unint8_first.npz'

# RESNET_MODEL = './resnet_101/resnet'
# PATCH_MODEL = None
# TOTAL_MODEL = None

RESNET_MODEL = None
PATCH_MODEL = './TFR_P20/model_TFR_P20-1140138'
TOTAL_MODEL = None

P_VALUE = 20
PRECROP_IMG_SIZE = [315, 315, 3]
TARGET_IMG_SIZE = [300, 300, 3]
LABEL_SIZE = [P_VALUE, P_VALUE, 2]
SINGLES = [2]

FLAGS = None

def run_infer(n_classes, from_tfr, in_memory_data=None, return_vals=False):

    graph = tf.get_default_graph()
    sess = tf.Session(graph=graph)

    if from_tfr:
        C_dataset = TFR_Dataset(1, FLAGS.infer_batch, FLAGS.prefetch_num, TEST_ROOT, 10000, TARGET_IMG_SIZE, TARGET_IMG_SIZE, LABEL_SIZE, SINGLES, file_shards=FLAGS.train_shards)
        C_dataset.make_tfr_dataset()

    else:
        C_dataset = Memory_Dataset(1, FLAGS.infer_batch, FLAGS.prefetch_num, UP_TEST, 2000, TARGET_IMG_SIZE, TARGET_IMG_SIZE, LABEL_SIZE, SINGLES)
        C_dataset.make_memory_dataset()
        C_dataset.make_memory_feed_dict(in_memory_data)

    fluid_iterator = Dataset_iterator(C_dataset.dataset)
    el = fluid_iterator.get_el()

    net = Recognition_Network(n_classes, 3, 0.05, sess, graph, P_VALUE)
    net.set_params(el)
    net.build_optimizer(lr=0.01,
                        resnet_path=None,
                        patch_model_path=PATCH_MODEL,
                        total_model_path=None)

    if from_tfr:
        sess.run(fluid_iterator.return_initialized_iterator(C_dataset.dataset))
    else:
        sess.run(fluid_iterator.return_initialized_iterator(C_dataset.dataset), feed_dict=C_dataset.feed_dict)

    pred_dict, labels_list = create_holders(n_classes)
    while True:
        try:
            scores, singles = net.score()
            for i, j in scores.items():
                pred_dict[i] = np.concatenate([pred_dict[i], j], axis=0)
            labels_list = np.concatenate([labels_list, singles], axis=0)

        except tf.errors.OutOfRangeError:
            break

    print('The metrics are: ')
    eval_metrics(pred_dict, labels_list, plot_auc=False)

    if return_vals:
        return pred_dict, labels_list

def run_train(det_save_path, n_classes, from_tfr):

    graph = tf.get_default_graph()
    sess = tf.Session(graph=graph)

    if from_tfr:
        C_dataset = TFR_Dataset(FLAGS.epochs, FLAGS.batch_size, FLAGS.prefetch_num, TRAIN_ROOT, 10000, TARGET_IMG_SIZE, TARGET_IMG_SIZE, LABEL_SIZE, SINGLES, file_shards=FLAGS.train_shards)
        C_dataset.make_tfr_dataset()
    else:
        C_dataset = Memory_Dataset(FLAGS.epochs, FLAGS.batch_size, FLAGS.prefetch_num, UP_TRAIN_300, 2000, TARGET_IMG_SIZE, TARGET_IMG_SIZE, LABEL_SIZE, SINGLES)
        C_dataset.make_memory_dataset(aug_crop=False)

    fluid_iterator = Dataset_iterator(C_dataset.dataset)
    el = fluid_iterator.get_el()

    net = Recognition_Network(n_classes, FLAGS.box_gamma, FLAGS.alpha_tail, sess, graph, P_VALUE, alpha_head=FLAGS.alpha_head)
    net.set_params(el)
    net.build_optimizer(lr=FLAGS.learning_rate,
                        resnet_path=RESNET_MODEL,
                        patch_model_path=PATCH_MODEL,
                        total_model_path=TOTAL_MODEL)

    step = 0
    if from_tfr:
        net.sess.run(fluid_iterator.return_initialized_iterator(C_dataset.dataset))
    else:
        C_dataset.make_memory_feed_dict()
        net.sess.run(fluid_iterator.return_initialized_iterator(C_dataset.dataset), feed_dict=C_dataset.feed_dict)

    train_start_time = time.time()
    interval_time = train_start_time

    try:
        while True:
            _, losses = net.sess.run([net.train_op, net.losses])

            if step % FLAGS.print_step == 0: # ~46000 cases / batch size of 9 ~= 5e3 steps per epoch
                # ~ 1800 / 6 ~= 300 steps per epoch for UP cases
                c_end = time.time()
                from_start = (c_end - train_start_time) / 3600 # minutes and seconds to hours
                from_last_step = (c_end - interval_time) / 60 # seconds to hours
                print('Step {0:g}, error {1:.3f}, {2:.2f} hrs from start, {3:.2f} mins from last epoch'.format(step, losses['total'], from_start, from_last_step))
                interval_time = c_end

            step += 1
            if step % FLAGS.save_step == 0: # 5000 steps per epochs * 50 epochs = 2.5e5 steps
                # 300 steps/epoch * 300 epochs for UP cases = 9e4 steps, just save only at end
                net.total_model_saver.save(net.sess, det_save_path, global_step=step)

    except tf.errors.OutOfRangeError:
        net.total_model_saver.save(net.sess, det_save_path, global_step=step)
        final_timing = (time.time() - train_start_time) / 3600
        print('Done training after total time of {0:.1f} hours'.format(final_timing))
        print('L2 loss tensors are: {}'.format(graph.get_collection('l2_loss')))

    # Scoring; no longer in training with BN
    if from_tfr:
        # need to reset C_dataset to change epoch
        C_dataset = TFR_Dataset(1, FLAGS.infer_batch, FLAGS.prefetch_num, TRAIN_ROOT, 10000, TARGET_IMG_SIZE, TARGET_IMG_SIZE, LABEL_SIZE, SINGLES, file_shards=FLAGS.train_shards)
        C_dataset.make_tfr_dataset()
        net.sess.run(fluid_iterator.return_initialized_iterator(C_dataset.dataset))
    else:
        C_dataset = Memory_Dataset(1, FLAGS.infer_batch, FLAGS.prefetch_num, UP_TRAIN_300, 2000, TARGET_IMG_SIZE, TARGET_IMG_SIZE, LABEL_SIZE, SINGLES)
        C_dataset.make_memory_dataset()
        C_dataset.make_memory_feed_dict()
        net.sess.run(fluid_iterator.return_initialized_iterator(C_dataset.dataset), feed_dict=C_dataset.feed_dict)

    pred_dict, labels_list = create_holders(n_classes)
    while True:
        try:
            scores, singles = net.score()
            for i, j in scores.items():
                pred_dict[i] = np.concatenate([pred_dict[i], j], axis=0)
            labels_list = np.concatenate([labels_list, singles], axis=0)

        except tf.errors.OutOfRangeError:
            break

    print('The training metrics are: ')
    eval_metrics(pred_dict, labels_list)

    #Scoring for test set;
    if from_tfr:
        C_dataset = TFR_Dataset(1, FLAGS.infer_batch, FLAGS.prefetch_num, TEST_ROOT, 10000, TARGET_IMG_SIZE, TARGET_IMG_SIZE, LABEL_SIZE, SINGLES, file_shards=FLAGS.train_shards)
        C_dataset.make_tfr_dataset()
        net.sess.run(fluid_iterator.return_initialized_iterator(C_dataset.dataset))
    else:
        C_dataset = Memory_Dataset(1, FLAGS.infer_batch, FLAGS.prefetch_num, UP_TEST, 2000, TARGET_IMG_SIZE, TARGET_IMG_SIZE, LABEL_SIZE, SINGLES)
        C_dataset.make_memory_dataset()
        C_dataset.make_memory_feed_dict()
        net.sess.run(fluid_iterator.return_initialized_iterator(C_dataset.dataset), feed_dict=C_dataset.feed_dict)

    pred_dict, labels_list = create_holders(n_classes)
    while True:
        try:
            scores, singles = net.score()
            for i, j in scores.items():
                pred_dict[i] = np.concatenate([pred_dict[i], j], axis=0)
            labels_list = np.concatenate([labels_list, singles], axis=0)

        except tf.errors.OutOfRangeError:
            break

    print('The Testing metrics are: ')
    eval_metrics(pred_dict, labels_list)

def main(inferring, jaccard):

    det_save_path = 'UP_noAUG_P{}/model_UP_noAUG_P{}'.format(P_VALUE, P_VALUE)
    print('Current params: ', det_save_path)
    print('From tfr flag is: ', FLAGS.from_tfr) # Default is False

    if inferring: # default is False
        run_infer(FLAGS.num_classes, FLAGS.from_tfr)
    else:
        run_train(det_save_path, FLAGS.num_classes, FLAGS.from_tfr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0002
    )
    parser.add_argument(
       '--epochs',
        type=int,
        default=2000
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=6
    )
    parser.add_argument(
        '--box_gamma',
        type=int,
        default=7
    )
    parser.add_argument(
        '--alpha_tail',
        type=float,
        default=0.05
    )
    parser.add_argument(
        '--alpha_head',
        type=float,
        default=None
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=2
    )
    parser.add_argument(
        '--prefetch_num',
        type=int,
        default=1
    )
    parser.add_argument(
        '--train_shards',
        type=int,
        default=20
    )
    parser.add_argument(
        '--test_shards',
        type=int,
        default=20
    )
    parser.add_argument(
        '--infer_batch',
        type=int,
        default=150
    )
    parser.add_argument(
        '--print_step',
        type=int,
        default=300
    )
    parser.add_argument(
        '--save_step',
        type=int,
        default=2.5e5
    )
    parser.add_argument(
        '--inferring',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--from_tfr',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--jaccard',
        type=bool,
        default=False
    )

    FLAGS, _ = parser.parse_known_args()
    main(FLAGS.inferring, FLAGS.jaccard)
