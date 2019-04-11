import tensorflow as tf
import numpy as np

from utils import Data, Data_indexed, truncated_Data

class Dataset_iterator:
    def __init__(self, dataset):
        self.structured_iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

    def get_el(self):
        return self.structured_iterator.get_next()

    def return_initialized_iterator(self, dataset):
        return self.structured_iterator.make_initializer(dataset)


#DO NOT put if statements inside the map functions, as they will result in issues with the iterator
class Dataset_Class:
    def __init__(self, epochs, batch_sz, prefetch_n, file_name, shuffle_buff, inp_img_size, out_img_size, label_size,
                 singles_size, preprocess, indefinite_repeat, truncated):
        self.epochs = epochs
        self.batch_sz = batch_sz
        self.prefetch_n = prefetch_n
        self.file_name = file_name
        self.shuffle_buff = shuffle_buff
        self.inp_img_size = inp_img_size
        self.out_img_size = out_img_size
        self.label_size = label_size
        self.singles_size = singles_size
        self.preprocess = preprocess
        self.indefinite_repeat = indefinite_repeat
        self.truncated = truncated

    def standardize_data(self, datatup):
        images = datatup.images
        labels = datatup.labels
        nus = datatup.nus
        singles = datatup.singles

        print('Inception preprocessing')
        images = (images - tf.reduce_min(images)) * (1.0 - 0.0) / (tf.reduce_max(images) - tf.reduce_min(images)) + 0.0
        images = (images - 0.5) * 2 # get values [-1, 1].  Shift and scale. For Resnet V2 and all tf models.

        return Data(images, labels, nus, singles)

    def truncate_data(self, datatup):
        images = datatup.images
        labels = datatup.labels
        nus = datatup.nus
        singles = datatup.singles

        return truncated_Data(images, singles)


class TFR_Dataset(Dataset_Class):
    # Old training had preprocess and augment as False, now they are True
    def __init__(self, epochs, batch_size, prefetch_n, file_name, shuffle_buff, inp_img_size, out_img_size, label_size,
                 singles_size, preprocess=True, indefinite_repeat=False, truncated=False, file_shards=20, augment=True):
        super().__init__(epochs, batch_size, prefetch_n, file_name, shuffle_buff, inp_img_size, out_img_size, label_size,
                         singles_size, preprocess, indefinite_repeat, truncated)
        self.file_shards = file_shards
        self.augment = augment

    def parse_tfr(self, exp_proto):
        features = {
            'image_raw': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
            'nu': tf.FixedLenFeature([], tf.int64),
            'singles_raw': tf.FixedLenFeature([], tf.string)
        }
        psed = tf.parse_single_example(exp_proto, features)

        image = tf.decode_raw(psed['image_raw'], tf.float32)
        image = tf.reshape(image, self.out_img_size)
        label = tf.decode_raw(psed['label_raw'], tf.float32)
        label = tf.reshape(label, self.label_size)
        nu = tf.cast(psed['nu'], tf.float32)
        single = tf.decode_raw(psed['singles_raw'], tf.float32)
        single = tf.reshape(single, self.singles_size)

        return Data(image, label, nu, single) #tf Dataset knows each tuple item is a component that needs to be batched together.  Each Data tuple is an element.

    def tfr_augment(self, datatup):
        image = datatup.images
        label = datatup.labels
        nu = datatup.nus
        single = datatup.singles

        print('TFR augmentation is on')
        image = tf.image.random_brightness(image, 0.3)
        image = tf.image.random_contrast(image, 0.7, 1.3)

        def flipper(m_img, m_label):
            m_img = tf.image.flip_left_right(m_img)
            m_label = tf.image.flip_left_right(m_label)

            return m_img, m_label

        def no_change(x, y):
            return x, y

        c_rand = tf.random_uniform([1])[0]
        image, label = tf.cond(c_rand < 0.5, lambda: no_change(image, label), lambda: flipper(image, label))

        return Data(image, label, nu, single)

    def shuf_rep_map_batch_fetch(self, dataset, buffer_size, epochs, batch_sz, map_func):
        if self.indefinite_repeat:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=buffer_size))

        elif epochs > 1:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=buffer_size, count=epochs))
            #ok to shuffle after parallel-interleave and before mapping.

        # Old TF bug: Cannot use new map_func (even if lambda) if tf.dataset not previously declared/created before tf Session started
        if map_func is not None:
            dataset = dataset.map(map_func, num_parallel_calls=2)

        if self.augment:
            dataset = dataset.map(self.tfr_augment, num_parallel_calls=2)

        dataset = dataset.batch(batch_sz)

        if self.preprocess:
            dataset = dataset.map(self.standardize_data)

        if self.truncated:
            dataset = dataset.map(self.truncate_data)

        dataset = dataset.prefetch(self.prefetch_n)

        return dataset

    def make_tfr_dataset(self):
        dataset = tf.data.Dataset.list_files(self.file_name, shuffle=True) #file names shuffled at every repeat
        dataset = dataset.apply(tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=self.file_shards, sloppy=True, block_length=1,
            buffer_output_elements=None, prefetch_input_elements=None))
        # interleave takes the files and converts them to TFRecordDataset, with block length determining number of elements to get
        # from each file (cycle-length).  That's why the shuffle function can be placed after the interleave, since interleave just
        # creates a feed of bytes elements, with map working on each of those elements.
        self.dataset = self.shuf_rep_map_batch_fetch(dataset, self.shuffle_buff, self.epochs, self.batch_sz, self.parse_tfr)

class Memory_Dataset(Dataset_Class):
    def __init__(self, epochs, batch_sz, prefetch_n, file_name, shuffle_buff, inp_img_size, out_img_size, label_size, singles_size,
                 preprocess=True, indefinite_repeat=False, truncated=False):
        super().__init__(epochs, batch_sz, prefetch_n, file_name, shuffle_buff, inp_img_size, out_img_size, label_size, singles_size,
                         preprocess, indefinite_repeat, truncated)
        self.images_PH = tf.placeholder(np.float32, [None] + inp_img_size)
        self.labels_PH = tf.placeholder(np.float32, [None] + label_size)
        self.nus_PH = tf.cast(tf.placeholder(np.int64, [None]), np.float32) # Memory cases stored as int64, so cast for Tensor and float arithmetic
        self.singles_PH = tf.placeholder(np.float32, [None] + singles_size)
        self.pd_indices = tf.placeholder(np.int32, [None])

    # MUST do BEFORE batching
    def augment(self, Data_tup, offset_value=15, label_cutt_off=6, class_of_interest=1):

        def flipper(m_img, m_label):
            m_img = tf.image.flip_left_right(m_img)
            m_label = tf.image.flip_left_right(m_label)

            return m_img, m_label

        def no_change(x, y):
            return x, y

        def label_mod(f_label_arr, f_lab): #f_lab is 0 or 1 from initial label
            class_0_labels = f_label_arr[..., 0] #unchanged from initial
            far_right_side = f_label_arr[:, :(label_cutt_off - 1), class_of_interest] # unchanged
            new_right_paracentral = tf.ones([f_label_arr.shape[0], 1]) * f_lab #either 0 or 1 based on f_lab
            new_left_paracentral = tf.ones([f_label_arr.shape[0], 1]) * f_lab #either 0 or 1 based on f_lab
            far_left_side = f_label_arr[:, (label_cutt_off + 1):, class_of_interest] #unchanged
            class_1_labels = tf.concat([far_right_side, new_right_paracentral, new_left_paracentral, far_left_side], axis=1)
            compiled_label = tf.stack([class_0_labels, class_1_labels], axis=-1)
            return compiled_label

        img  = Data_tup.images
        label = Data_tup.labels
        nu = Data_tup.nus
        single = Data_tup.singles

        img = tf.image.random_brightness(img, 0.3)
        img = tf.image.random_contrast(img, 0.7, 1.3)

        c_rand = tf.random_uniform([1])[0]
        img, label = tf.cond(c_rand < 0.5, lambda: no_change(img, label), lambda: flipper(img, label))

        right_label = label[0, label_cutt_off - 1, class_of_interest] #copy since int is immutable
        left_label = label[0, label_cutt_off, class_of_interest]

        H_off = tf.cast((tf.random_uniform([1]) * offset_value), tf.int32)[0]
        W_off = tf.cast((tf.random_uniform([1]) * offset_value), tf.int32)[0]
        label = tf.cond(W_off < 3, lambda: label_mod(label, right_label), lambda: label)
        label = tf.cond(W_off > 12, lambda: label_mod(label, left_label), lambda: label)

        img = tf.image.crop_to_bounding_box(img, H_off, W_off, self.out_img_size[0], self.out_img_size[1])

        return Data(img, label, nu, single)

    def shuf_rep_map_batch_fetch(self, dataset, buffer_size, epochs, batch_sz, map_func):
        if self.indefinite_repeat:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=buffer_size))

        elif epochs > 1:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=buffer_size, count=epochs))
            #ok to shuffle after parallel-interleave and before mapping.

        # TF bug: Cannot use new map_func (even if lambda) if tf.dataset not previously declared/created before tf Session started.
        if map_func is not None:
            dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=map_func, batch_size=batch_sz, num_parallel_calls=2))
        else:
            dataset = dataset.batch(batch_sz)

        if self.preprocess:
            dataset = dataset.map(self.standardize_data)

        if self.truncated:
            dataset = dataset.map(self.truncate_data)

        dataset = dataset.prefetch(self.prefetch_n)

        return dataset

    def make_memory_dataset(self, aug_crop=False):

        dataset = tf.data.Dataset.from_tensor_slices(Data(self.images_PH, self.labels_PH, self.nus_PH, self.singles_PH))
        c_map_func = None
        if aug_crop:
            c_map_func = self.augment

        self.dataset = self.shuf_rep_map_batch_fetch(dataset, self.shuffle_buff, self.epochs, self.batch_sz, c_map_func)

    def make_short_dataset(self):
        self.dataset = tf.data.Dataset.from_tensor_slices(Data(self.images_PH, self.labels_PH, self.nus_PH, self.singles_PH))

    def make_memory_feed_dict(self, in_memory_Data=None):
        if in_memory_Data is not None:
            c_data = in_memory_Data # if you just process the files and keep the Data tuple in memory

        else:
            with np.load(self.file_name) as c_file:
                c_data = Data(c_file['images'], c_file['labels'], c_file['nus'], c_file['singles'])

        self.feed_dict = {self.images_PH: c_data.images,
                          self.labels_PH: c_data.labels,
                          self.nus_PH: c_data.nus,
                          self.singles_PH: c_data.singles}

    def make_indexed_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(
            Data_indexed(self.images_PH, self.labels_PH, self.nus_PH, self.singles_PH, self.pd_indices))
        self.dataset = dataset.batch(self.batch_sz)

    def make_indexed_feed_dict(self):
        with np.load(self.file_name) as c_file:
            c_data = Data_indexed(c_file['images'], c_file['labels'], c_file['nus'], c_file['singles'], c_file['pd_indices'])

        self.feed_dict = {self.images_PH: c_data.images,
                          self.labels_PH: c_data.labels,
                          self.nus_PH: c_data.nus,
                          self.singles_PH: c_data.singles,
                          self.pd_indices: c_data.pd_indices}

def simple_load_npz(filename):
    with np.load(filename) as c_file:
        c_data = Data_indexed(c_file['images'], c_file['labels'], c_file['nus'], c_file['singles'],
                              c_file['pd_indices'])

    return c_data

def single_tfr_read(filename, sess, out_img_size=[300, 300, 3], label_size=[12, 12, 2], singles_size=[2]):
    def parse_tfr(exp_proto):
        features = {
            'image_raw': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
            'nu': tf.FixedLenFeature([], tf.int64),
            'singles_raw': tf.FixedLenFeature([], tf.string)
        }
        psed = tf.parse_single_example(exp_proto, features)

        image = tf.decode_raw(psed['image_raw'], tf.float32)
        image = tf.reshape(image, out_img_size)
        label = tf.decode_raw(psed['label_raw'], tf.float32)
        label = tf.reshape(label, label_size)
        nu = tf.cast(psed['nu'], tf.float32)
        single = tf.decode_raw(psed['singles_raw'], tf.float32)
        single = tf.reshape(single, singles_size)

        return Data(image, label, nu, single)

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_tfr)
    dataset = dataset.batch(200)
    d_iter = dataset.make_one_shot_iterator()
    el = d_iter.get_next()

    return sess.run(el)

def get_TFR_structured_iterator(sess, epochs, batch_size, file_name, shuffle_buff, inp_img_size, out_img_size, label_size,
                                singles_size, file_shards, indefinite_repeat=True, return_initalizer=False):

    tfr_dataset = TFR_Dataset(epochs, batch_size, 1, file_name, shuffle_buff, inp_img_size, out_img_size, label_size,
                              singles_size, indefinite_repeat=indefinite_repeat, truncated=True, file_shards=file_shards)
    tfr_dataset.make_tfr_dataset()

    d_iterator = Dataset_iterator(tfr_dataset.dataset)
    iterator_init = d_iterator.return_initialized_iterator(tfr_dataset.dataset)
    sess.run(iterator_init)

    if return_initalizer:
        return d_iterator.structured_iterator, iterator_init
    else:
        return d_iterator.structured_iterator

def get_npz_structured_iterator(sess, epochs, batch_size, file_name, shuffle_buff, inp_img_size, out_img_size, label_size,
                                singles_size, indefinite_repeat=True, return_initalizer=True, aug_crop=True):

    mem_dataset = Memory_Dataset(epochs, batch_size, 1, file_name, shuffle_buff, inp_img_size, out_img_size, label_size, singles_size, indefinite_repeat=indefinite_repeat, truncated=True)
    mem_dataset.make_memory_dataset(aug_crop=aug_crop)
    mem_dataset.make_memory_feed_dict()

    d_iterator = Dataset_iterator(mem_dataset.dataset)
    iterator_init = d_iterator.return_initialized_iterator(mem_dataset.dataset)
    sess.run(iterator_init, feed_dict=mem_dataset.feed_dict)

    if return_initalizer:
        return d_iterator.structured_iterator, iterator_init, mem_dataset
    else:
        return d_iterator.structured_iterator