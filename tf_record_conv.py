#!/usr/bin/env python
from check_warnings import suppress_warnings, check_list
suppress_warnings()

import tensorflow as tf
import argparse
import math
from tqdm import tqdm
import os

from process import construct_images_labels, load_pandas, Data


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def data_buff(data_tup, writer):

    images = data_tup.images
    labels = data_tup.labels
    nus = data_tup.nus
    singles = data_tup.singles

    sample_num = images.shape[0]
    for i in range(sample_num):

        img_raw = images[i].tostring()
        label_arr_raw = labels[i].tostring()
        singles_raw = singles[i].tostring()

        a_features = tf.train.Features(feature={
            'image_raw': bytes_feature(img_raw),
            'label_raw': bytes_feature(label_arr_raw),
            'nu': int64_feature(nus[i]),
            'singles_raw': bytes_feature(singles_raw)
        })

        example = tf.train.Example(features=a_features)

        writer.write(example.SerializeToString())

def tf_converter(i_df, file_root, batch_size, shards, P_value):

    cases_per_shard = int(math.ceil(i_df.shape[0] / shards))
    batches_per_shard = int(math.ceil(cases_per_shard / batch_size))
    counter = 0
    shard_flag = True

    for single_shard in range(shards):
        if not shard_flag:
            break
        file_name = file_root + str(single_shard) + '.tfr'

        with tf.python_io.TFRecordWriter(file_name) as writer:
            for i in tqdm(range(batches_per_shard), total=batches_per_shard, desc=file_name, unit='batches'):
            #for i in range(t_num):
            #    if (i % 20 == 0):
            #        print('Progress {}/total'.format(i, t_num))
                try:
                    c_df = i_df[counter*batch_size: (counter+1)*batch_size]
                    counter += 1
                    a_data = construct_images_labels(c_df, P_value)
                    data_buff(a_data, writer)
                except IndexError:
                    # IndexError is from trying to index axis 3 in construct_images_label fxn when c_df is empty from indexing beyond the df size
                    print('End of indices at shard #{} at batch #{}'.format(single_shard, i))
                    if i == 0: # if batch is 0, then the file is empty
                        os.remove('./' + file_name)
                        print('Empty file removed: ', file_name)
                    shard_flag = False
                    break



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=150
    )
    parser.add_argument(
        '--shards',
        type=int,
        default=20
    )
    parser.add_argument(
        '--P_value',
        type=int,
        default=12
    )
    args, _ = parser.parse_known_args()

    train_df, test_df = load_pandas()
    train_df = train_df.sample(frac=1)
    test_df = test_df.sample(frac=1) #TODO: no need to shuffle test set??

    train_root = 'INCEP_train_P'
    test_root = 'INCEP_test_P'

    if not os.path.exists(train_root + str(args.P_value)):
        os.mkdir(train_root + str(args.P_value))
    if not os.path.exists(test_root + str(args.P_value)):
        os.mkdir(test_root + str(args.P_value))

    tf_converter(train_df, '{}/train_shard_'.format(train_root + str(args.P_value)), args.batch_size, args.shards, args.P_value)
    tf_converter(test_df, '{}/test_shard_'.format(test_root + str(args.P_value)), args.batch_size, args.shards, args.P_value)

if __name__ == '__main__':
    main()
