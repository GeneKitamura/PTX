import numpy as np
import cv2
import pandas as pd
import math
import re
import sklearn

from skimage.transform import resize

from utils import path_dict, Data

# In original paper, scale images between [-1 , 1] and resize to (299, 299).  Resnet 101 size is always ceil(h or w / 32).
# For us, will be [-1, 1] for inception or mean subtraction for resnet preprocessing and (300, 300) image size for ease.

def process_test_boxes(P_value):
    df = load_pandas(test_box_path='./test_boxes.h5')
    test_boxes_Data = construct_images_labels(df, P_value)

    return test_boxes_Data

def construct_images_labels(inp_df, P_value, out_img_size=(300, 300, 3), n_classes=2, img_path='./../CXR/images/', preprocess='inception', f_tfrecord=True):
    batch_input = inp_df.copy()
    batch_input['img'] = batch_input['Image Index'].map(lambda x: img_path + x) #pandas series

    batch_input[['x', 'y', 'h', 'w']] = batch_input[['x', 'y', 'h', 'w']] * (P_value - 1) # arrays start at 0
    batch_input['xmax'] = batch_input['x'] + batch_input['w']
    batch_input['ymax'] = batch_input['y'] + batch_input['h']
    batch_input = batch_input.rename(columns={'x': 'xmin', 'y': 'ymin'})
    # if a border is touching a box, it's included.  So min the min and max the max.
    batch_input[['xmin', 'ymin']] = batch_input[['xmin', 'ymin']].applymap(lambda x: math.floor(x))
    batch_input[['xmax', 'ymax']] = batch_input[['xmax', 'ymax']].applymap(lambda x: math.ceil(x))
    batch_input = batch_input.drop(['h', 'w'], axis=1)

    images_list = []
    labels_list = []
    nu_list = []
    single_labels = []

    for idx, row in batch_input.iterrows():
        _img, _label, _xmin, _xmax, _ymin, _ymax, _nu, _multi = batch_input.loc[idx, ['img', 'label', 'xmin', 'xmax', 'ymin', 'ymax', 'nu', 'multi_labels']]
        _IP_img = cv2.imread(_img) #images of dtype uint8
        # identical images are stacked along axis=3
        _IP_img = resize(_IP_img, out_img_size, mode='reflect') #img_as_float
        images_list.append(_IP_img)
        _label_arr = np.zeros((P_value, P_value, n_classes))

        if n_classes==2:
            single_labels.append([_label]) # make into list for consistency with multi
        if n_classes==15:
            single_labels.append(_multi)

        # For unbounded cases
        if _xmax == 0: #cannot be xmin since the box can start at xmin of 0
            if n_classes==2:
                _label_arr[..., _label] = 1
            if n_classes==15:
                for i in _multi:
                    _label_arr[..., i] = 1
        # For boxed cases
        else:
            _xmax += 1
            _ymax += 1
            _label_arr[_ymin: _ymax, _xmin: _xmax, _label] = 1
        labels_list.append(_label_arr)
        nu_list.append(_nu)

    images_list = np.array(images_list, np.float32)

    preprocess_methods = {'resnet', 'inception', 'none'}
    if preprocess not in preprocess_methods:
        raise KeyError('Must select a preprocess method from {}'.format(preprocess_methods))
    if preprocess == 'inception':
        images_list = (images_list - 0.5) * 2 # get values [-1, 1].  Shift and scale.

    if preprocess == 'resnet': #substract mean instead for Resnet preprocessing
        # RGB values from tensorflow resnet divded by uint8 pixel range (255)
        images_list[..., 0] -= 123.68/255
        images_list[..., 1] -= 116.78/255
        images_list[..., 2] -= 103.94/255

    labels_list = np.array(labels_list, np.float32)
    nu_list = np.array(nu_list, np.float32)

    # single labels as one-hot
    single_labels = sklearn.preprocessing.MultiLabelBinarizer(np.arange(n_classes)).fit_transform(single_labels)

    if f_tfrecord: #formatted for tfrecord serializing
        single_labels = np.array(single_labels, np.float32)
        nu_list = nu_list.astype(np.int64) # will be casted to float32 using during tf record dataset map
        #image list, labels, and single_labels list as float32, and nu as int64

    return Data(images_list, labels_list, nu_list, single_labels)


def load_pandas(path_to_hdf='./data.h5', combined=True, test_box_path=None, use_all=False):
    # current data.h5 has all non-PTX as nu of 1
    # Just to get the box cases
    if test_box_path is not None:
        with pd.HDFStore(test_box_path, mode='r') as store:
            return store.get('cases')

    with pd.HDFStore(path_to_hdf, mode='r') as store:
        comb_train_path = store.get('train_ptx') #4696
        comb_test_path = store.get('test_ptx') #541
        other_training_use = store.get('other_train_use') #41946
        other_testing_use = store.get('other_test_use') # 11120
        other_training_hold = store.get('other_train_hold') #41941
        other_testing_hold = store.get('other_test_hold') #11094

    if use_all:
        return pd.concat([comb_train_path, other_training_use, other_training_hold], axis=0, sort=False), \
               pd.concat([comb_test_path, other_testing_use, other_testing_hold], axis=0, sort=False)
    elif combined: # concat sort is for sorting the column titles in alphabetical order
        return pd.concat([comb_train_path, other_training_use], axis=0, sort=False), pd.concat([comb_test_path, other_testing_use], axis=0, sort=False)
    else:
        return comb_train_path, comb_test_path, other_training_use, other_testing_use



def split_all_cases():
    # TODO: create function to split all cases and boxes into train and test sets.
    pass


def split_ptx(train_file='./chest_8/train_val_list.txt', test_file='./chest_8/test_list.txt', save=True, equal_check_sample=False, test_box_save=False):

    path, boxes, other = csv_pandas()

    if equal_check_sample:
        print('is not saved into h5, just returned')
        path = path.sample(frac=1)
        boxes = boxes.sample(frac=1)
        other = other.sample(frac=1)
        return pd.concat([boxes[:30], path[:30], other[:30]], axis=0).fillna(0)

    other = other.assign(x=0.0, y=0.0, h=0.0, w=0.0)

    train_list = pd.read_table(train_file, sep='\n', names=['Image Index'])
    test_list = pd.read_table(test_file, sep='\n', names=['Image Index'])

    # PTX cases
    path_training = path.reset_index().merge(train_list, how='inner', on='Image Index').set_index('index').sort_values('Image Index')
    path_testing = path.reset_index().merge(test_list, how='inner', on='Image Index').set_index('index').sort_values('Image Index') #make sure it's sorted
    path_training = pd.concat([path_training, path_testing[:1982]]) #manually adding more training cases.  Make sure same patient with multiple images NOT split up.  Index cut off of 26000.
    path_testing = path_testing[1982:].copy() # ends up with 520 cases

    # Box cases - Remember the cases are organized by FINDING, so index is NOT ordered when sorted by "Image Index"
    box_training = boxes.reset_index().merge(train_list, how='inner', on='Image Index').set_index('index').sort_values("Image Index") # ends up with 0 training cases
    box_testing = boxes.reset_index().merge(test_list, how='inner', on='Image Index').set_index('index').sort_values("Image Index") # all cases are test
    box_training = box_testing[:77].copy() # Box cases already exclusive to ptx cases.  Just make sure patients with multiple images don't get split up.  Index cutoff of 26000.
    box_testing = box_testing[77:].copy() # end up with 21 cases

    # Other cases:
    other_training = other.reset_index().merge(train_list, how='inner', on='Image Index').set_index('index') #83887 cases
    other_testing = other.reset_index().merge(test_list, how='inner', on='Image Index').set_index('index') #22214 cases
    half_other_training = int(len(other_training)/2) + 3 #move all images from one patient into one half
    half_other_testing = int(len(other_testing)/2) + 13 # results in no split patient cases
    other_training_use = other_training[:half_other_training] # to use; findings are random when cases sorted by Image Index
    other_training_hold = other_training[half_other_training:] # to save
    other_testing_use = other_testing[:half_other_testing] # to use
    other_testing_hold = other_testing[half_other_testing:] # to save

    comb_train_path = pd.concat([path_training, box_training], axis=0, sort=False).fillna(0)
    comb_test_path = pd.concat([path_testing, box_testing], axis=0, sort=False).fillna(0)

    if save:
        # There is index overlap between the non-boxes and boxes cases
        print('SAVING TO HDFstore')
        with pd.HDFStore('./data.h5', mode='w') as store:
            store.put('train_ptx', comb_train_path, format='f')
            store.put('test_ptx', comb_test_path, format='f')
            store.put('other_train_use', other_training_use, format='f')
            store.put('other_test_use', other_testing_use, format='f')
            store.put('other_train_hold', other_training_hold, format='f')
            store.put('other_test_hold', other_testing_hold, format='f')

    if test_box_save:
        print('saving test box cases')
        with pd.HDFStore('./test_boxes.h5', mode='w') as store:
            store.put('cases', box_testing, format='f')

    return comb_train_path, comb_test_path, other_training_use, other_testing_use, other_training_hold, other_testing_hold


def csv_pandas(data_csv='./chest_8/Data_Entry_2017.csv', box_csv='./chest_8/BBox_List_2017.csv', finding='Pneumothorax', img_size=(1024, 1024), binary_classification=True):

    data_df = pd.read_csv(data_csv)
    data_df = data_df.loc[:, ['Image Index', 'Finding Labels']]

    # Note the column of Finding Label WITHOUT the plural 's' compared to data_df
    # Above point is due to many of the labels being on the SAME image.
    # index based on finding and NOT on Image Index
    box_df = pd.read_csv(box_csv)
    box_df = box_df.rename(columns={'Bbox [x': 'x', 'h]': 'h'}).loc[:, ['Image Index', 'Finding Label', 'x', 'y', 'h', 'w']]
    box_df['Finding Label'] = box_df.apply(lambda row: re.sub(r'Inf.*', 'Infiltration', row['Finding Label']), axis=1)

    # Making the sizes proportional: Remember image is W x H by convention when only 2 dimensional
    box_df[['x', 'w']] =  box_df[['x', 'w']] / img_size[0]
    box_df[['y', 'h']] =  box_df[['y', 'h']] / img_size[1]
    #TODO: if img_size variable, have img size in csv, and extract as column

    # Extract box cases from non-boxes cases and reindex
    merged_all = data_df.reset_index().merge(box_df, how='outer', on='Image Index', indicator=True).rename(columns={'index': 'org_idx'})
    #Checking to see whether all box cases are in the the data list
    assert (merged_all[merged_all['_merge'] == 'right_only'].shape[0] == 0), "Some box Image Index NOT in data_df"

    # Re-indexed cases with preserved original index in new column
    data_df = merged_all[merged_all['_merge'] == 'left_only'].drop(['x', 'y', 'w', 'h', '_merge', 'Finding Label'], axis=1)
    box_df = merged_all[merged_all['_merge'] != 'left_only'].drop(['_merge', 'Finding Labels'], axis=1)

    data_df['nu'] = 0
    box_df['nu'] = 1

    path2id, id2path = path_dict()

    data_df['multi_labels'] = data_df.apply(lambda row: [path2id[i] for i in row['Finding Labels'].split('|')], axis=1)
    data_df['multi_labels'] = data_df['multi_labels'].map(np.array)

    box_df['multi_labels'] = box_df.apply(lambda row: [path2id[row['Finding Label']]], axis=1)
    box_df['multi_labels'] = box_df['multi_labels'].map(np.array)
    box_df = box_df.rename(columns={'Finding Label': 'Finding Labels'})

    # Just assigning label column now for all rows
    data_df['label'] = data_df['multi_labels']
    box_df['label'] = box_df['multi_labels']

    if binary_classification:
        # boolean filtering results in COPY of df or np array
        path_filter = data_df['Finding Labels'].str.contains(finding)
        path_cases = data_df.loc[path_filter].copy()
        non_path_cases = data_df.loc[~path_filter].copy()

        # Selecting out the finding of interest
        # Remember many images actually have MULTIPLE boxes
        path_boxes = box_df.loc[box_df['Finding Labels'] == finding].copy()
        ddx_boxes = box_df.loc[box_df['Finding Labels'] != finding].copy()

        # Assigning label for binary study
        path_cases['label'] = 1
        path_boxes['label'] = 1
        non_path_cases['label'] = 0

        # All patches should be label 0 for non_path cases
        non_path_cases['nu'] = 1

        return path_cases, path_boxes, non_path_cases

    else:

        return data_df, box_df

