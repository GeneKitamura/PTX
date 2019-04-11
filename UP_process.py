#!/usr/bin/env python
from check_warnings import suppress_warnings, check_list
suppress_warnings()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import math
import cv2
import pydicom
import os
import argparse
import skimage

from skimage import transform, exposure

from utils import Data, Data_indexed, rescale, erase_borders

FLAGS = None

xmin_dict = {'right': 0, 'left': 1, 'bilateral': 0}
xmax_dict = {'right': 1, 'left': 2, 'bilateral': 2}
lat_or_no_image_indices = [7740, 7944, 7971, 8437, 10912, 11510, 6997, 8234, 8352, 8834, 9023, 10542, 10968, 11229, 12209, 12323]
bad_positioning = [471, 4119, 4957, 8555, 8774, 8285, 9967, 10148, 10198, 10281, 10939, 11072, 11073, 12128, 9502, 10104, 10255, 10351, 10432, 10485, 10943, 11135, 11139, 11386, 11478, 11609, 11763, 11972, 11979, 12194]

def read_UP_excel(excel_file='./kolo.xlsx', cal_or_val='calibration', P=20, describe_it=True, large_cases_only=True):
    descriptions = None

    total_df = pd.read_excel(excel_file)
    img_path = './../UP/decomp_UP_{}/'.format(cal_or_val)

    # Get rid of questionable cases
    total_df = total_df[total_df['SizeCategory'] != 'unknown']
    total_df = total_df[total_df['QualifierCategory'] != 'uncertain']

    # Get rid of unnecessary columns
    total_df = total_df.drop(['DeidModalityDeidStudyDate', 'FileIndex', 'DeidPatientName', 'DeidPatientID', 'Signature', 'study_id', 'QualifierCategory'], axis=1).fillna('na')

    #Manually getting rid of bad cases:
    total_df = total_df[~total_df.index.isin(lat_or_no_image_indices)]
    total_df = total_df[~total_df.index.isin(bad_positioning)]

    # Getting df ready to labelize
    half_point = math.floor(P/2)
    total_df = total_df.assign(xmin=0, xmax=0)
    total_df.loc[total_df['IsPositive'] == True, 'xmin'] = total_df.loc[total_df['IsPositive'] == True].apply(lambda row: xmin_dict[row['FinalSide']], axis=1)
    total_df.loc[total_df['IsPositive'] == True, 'xmax'] = total_df.loc[total_df['IsPositive'] == True].apply(lambda row: xmax_dict[row['FinalSide']], axis=1)
    total_df.loc[:, ['xmin', 'xmax']] *= half_point

    total_df['class_0_fill'] = ~total_df['IsPositive'] * 1
    total_df['label'] = total_df['IsPositive'] * 1

    # Select out cases labels as calibration and validation
    other_df = total_df[~total_df['DataSetCategory'].isin(['calibration', 'validation'])]
    total_df = total_df[total_df['DataSetCategory'].isin(['calibration', 'validation'])]

    #Modify columns
    total_df['dicom'] = total_df['DatasetFileName'].map(lambda x: img_path + x) #pandas series
    total_df = total_df.drop(['DatasetFileName'], axis=1)

    # Select out calibration and validation sets
    total_df = total_df[total_df['DataSetCategory'] == cal_or_val]

    # make sure file path exists
    total_df = total_df[total_df['dicom'].map(lambda x: os.path.exists(x))]

    if large_cases_only:
        # Size categories: small, significant, unknown, percent, tension
        total_df = total_df[~total_df['SizeCategory'].isin(['small', 'percent'])]

    if describe_it:
        descriptions = total_df.describe(include='all')[:4]

    return total_df # calibration or validation

def construct_UP(inp_df, output_size, n_classes=2, P=20, preprocess='inception', np_save_file=None, save_with_indices=None,
                 return_Data=False, log_save=False, conv_to_uint8=True, min_max_exposure=True):
    batch_input = inp_df.copy() # copy to prevent modifying original df

    images_list = []
    labels_list = []
    nu_list = []
    single_labels = []
    pd_indices = []

    LL_view_list = []
    no_view_list = []
    derivative_view = []

    for idx, row in batch_input.iterrows():
        _dicom, _label, _xmin, _xmax, class_0_fill = row[['dicom', 'label', 'xmin', 'xmax', 'class_0_fill']]

        a = pydicom.dcmread(_dicom)

        try:
            if a.DerivationDescription: #derivate view
                derivative_view.append(_dicom)
                continue

        except AttributeError:
            pass

        try:
            c_view = a.ViewPosition
            if c_view == 'LL':
                LL_view_list.append(_dicom)
                continue

            if c_view not in ['PA', 'AP']:
                no_view_list.append(_dicom)
                continue

        except AttributeError:
            continue

        _img = a.pixel_array
        bs = a.BitsStored

        _img = erase_borders(_img)

        if min_max_exposure:

            _img = rescale(_img, 0, 1, np.min(_img), np.max(_img)).astype(np.float32)

            if a.PhotometricInterpretation == 'MONOCHROME1':
                _img = skimage.util.invert(_img, signed_float=False)

            #All pixels ABOVE 99th percentile set TO 99th percentile (getting around super bright labels)
            _img[_img > np.percentile(_img, 99)] = np.percentile(_img, 99)

            #Rescaling again after pixel cutt-off
            _img = rescale(_img, 0, 1, np.min(_img), np.max(_img)).astype(np.float32)

            if conv_to_uint8:
                _img = skimage.img_as_uint(_img)

        else:
            _img = exposure.rescale_intensity(_img, in_range=('uint' + str(bs))) #most, if not all, are of type uint16

            if conv_to_uint8:
                _img = skimage.img_as_uint(_img)

            if a.PhotometricInterpretation == 'MONOCHROME1':
                _img = cv2.bitwise_not(_img)

        _img = transform.resize(_img, (output_size, output_size), mode='reflect', anti_aliasing=True) #img_as_float
        _img = np.stack([_img, _img, _img], axis=-1)
        images_list.append(_img)

        _label_arr = np.zeros((P, P, n_classes))

        single_labels.append([_label])
        _label_arr[..., 0] = class_0_fill

        # For boxed cases
        _label_arr[:, _xmin: _xmax, _label] = 1
        labels_list.append(_label_arr)

        #TODO: change the nu for PTX (0 or 1)
        #all chest cases have laterality, and negatives cases should have class 0 in all patches
        #nu_list.append(_label) # only positives cases with laterality known have nu of 1
        nu_list.append(1) # all cases have nu of 1.
        #nu_list.append(np.logical_not(_label)) # NL with nu of 1 and PTX with nu of 0
        # nu_list.append(0) # no cases with localization

        pd_indices.append(idx)

    images_list = np.array(images_list, np.float32)
    pd_indices = np.array(pd_indices, np.int32)

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

    single_labels = np.array(single_labels, np.float32)
    nu_list = nu_list.astype(np.int64)
    #image list, labels, and single_labels list as float32, and nu as int64

    if np_save_file is not None:
        np.savez(np_save_file, images=images_list, labels=labels_list, nus=nu_list, singles=single_labels)

    if save_with_indices is not None:
        np.savez(save_with_indices, images=images_list, labels=labels_list, nus=nu_list, singles=single_labels, pd_indices=pd_indices)

    if log_save:
        with open('LL_view.txt', 'w') as f:
            for i in LL_view_list:
                f.write(i + '\n')

        with open('no_view.txt', 'w') as f:
            for i in no_view_list:
                f.write(i + '\n')

        with open('derivative_view.txt', 'w') as f:
            for i in derivative_view:
                f.write(i + '\n')


    if return_Data:
        return Data_indexed(images_list, labels_list, nu_list, single_labels, pd_indices)

    #Before manual cleaning
    #Calibration set for large PTX: 698 Normals and 162 PTX
    #Validation set for large PTX: 1314 Normals and 48 PTX

    #After manual cleaning
    # Calibration large PTX: 682 NL and 159 PTX for 841 total.
    # Validation large PTX: 1287 NL and 48 PTX for 1335 total.


def main():
    # NO proprocessing of images, just img_to_float due to resizing.
    # standardization to [-1, 1] will be done after augmentation in the dataset pipeline.

    c_df = read_UP_excel(cal_or_val=FLAGS.cal_or_val, P=FLAGS.P_value)
    construct_UP(c_df, FLAGS.output_size, save_with_indices=FLAGS.save_with_indices, np_save_file=FLAGS.np_save_file,
                 P=FLAGS.P_value, conv_to_uint8=True, preprocess='inception', min_max_exposure=True) # make sure to pass in save path

def remix(cal_path=None, val_path=None, NL_cut=394, PTX_cut=42):

    # 1969 NL and 207 PTX for 2176 total cases.  ~ 10% are PTX.
    # Combined 1575 NL and 165 PTX for 1740 train (80%), and 394 NL and 42 PTX for 436 validation (20%).

    cal_df = read_UP_excel(cal_or_val='calibration', P=FLAGS.P_value)
    NL_cal_df = cal_df[cal_df['label'] == 0]
    NL_cal_data = construct_UP(NL_cal_df, FLAGS.output_size, P=FLAGS.P_value, return_Data=True)
    PTX_cal_df = cal_df[cal_df['label'] == 1]
    PTX_cal_data = construct_UP(PTX_cal_df, FLAGS.output_size, P=FLAGS.P_value, return_Data=True)

    val_df = read_UP_excel(cal_or_val='validation', P=FLAGS.P_value)
    NL_val_df = val_df[val_df['label'] == 0]
    NL_val_data = construct_UP(NL_val_df, FLAGS.output_size, P=FLAGS.P_value, return_Data=True)
    PTX_val_df = val_df[val_df['label'] == 1]
    PTX_val_data = construct_UP(PTX_val_df, FLAGS.output_size, P=FLAGS.P_value, return_Data=True)

    new_cal_NL_images = np.concatenate((NL_cal_data.images, NL_val_data.images[:-NL_cut]))
    new_val_NL_images = NL_val_data.images[-NL_cut:]
    new_cal_NL_labels = np.concatenate((NL_cal_data.labels, NL_val_data.labels[:-NL_cut]))
    new_val_NL_labels = NL_val_data.labels[-NL_cut:]
    new_cal_NL_nus = np.concatenate((NL_cal_data.nus, NL_val_data.nus[:-NL_cut]))
    new_val_NL_nus = NL_val_data.nus[-NL_cut:]
    new_cal_NL_singles = np.concatenate((NL_cal_data.singles, NL_val_data.singles[:-NL_cut]))
    new_val_NL_singles = NL_val_data.singles[-NL_cut:]
    new_cal_NL_indices = np.concatenate((NL_cal_data.pd_indices, NL_val_data.pd_indices[:-NL_cut]))
    new_val_NL_indices = NL_val_data.pd_indices[-NL_cut:]


    new_cal_PTX_images = np.concatenate((PTX_cal_data.images, PTX_val_data.images[:-PTX_cut]))
    new_val_PTX_images = PTX_val_data.images[-PTX_cut:]
    new_cal_PTX_labels = np.concatenate((PTX_cal_data.labels, PTX_val_data.labels[:-PTX_cut]))
    new_val_PTX_labels = PTX_val_data.labels[-PTX_cut:]
    new_cal_PTX_nus = np.concatenate((PTX_cal_data.nus, PTX_val_data.nus[:-PTX_cut]))
    new_val_PTX_nus = PTX_val_data.nus[-PTX_cut:]
    new_cal_PTX_singles = np.concatenate((PTX_cal_data.singles, PTX_val_data.singles[:-PTX_cut]))
    new_val_PTX_singles = PTX_val_data.singles[-PTX_cut:]
    new_cal_PTX_indices = np.concatenate((PTX_cal_data.pd_indices, PTX_val_data.pd_indices[:-PTX_cut]))
    new_val_PTX_indices = PTX_val_data.pd_indices[-PTX_cut:]

    new_cal_images = np.concatenate((new_cal_NL_images, new_cal_PTX_images))
    new_cal_labels = np.concatenate((new_cal_NL_labels, new_cal_PTX_labels))
    new_cal_nus = np.concatenate((new_cal_NL_nus, new_cal_PTX_nus))
    new_cal_singles = np.concatenate((new_cal_NL_singles, new_cal_PTX_singles))
    new_cal_indices = np.concatenate((new_cal_NL_indices, new_cal_PTX_indices))

    new_val_images = np.concatenate((new_val_NL_images, new_val_PTX_images))
    new_val_labels = np.concatenate((new_val_NL_labels, new_val_PTX_labels))
    new_val_nus = np.concatenate((new_val_NL_nus, new_val_PTX_nus))
    new_val_singles = np.concatenate((new_val_NL_singles, new_val_PTX_singles))
    new_val_indices = np.concatenate((new_val_NL_indices, new_val_PTX_indices))

    print('P size is: ', FLAGS.P_value)

    if cal_path is not None:
        np.savez(cal_path, images=new_cal_images, labels=new_cal_labels, nus=new_cal_nus, singles=new_cal_singles, pd_indices=new_cal_indices)

    if val_path is not None:
        np.savez(val_path, images=new_val_images, labels=new_val_labels, nus=new_val_nus, singles=new_val_singles, pd_indices=new_val_indices)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cal_or_val',
        type=str,
        choices=['calibration', 'calibration'],
        default='calibration')
    parser.add_argument(
        '--save_with_indices',
        type=str,
        default='./min_max_uint8_INCEP_315_kolo_calibration_p12.npz'
    )
    parser.add_argument(
        '--np_save_file',
        type=str,
        default=None
    )
    parser.add_argument(
        '--output_size',
        type=int,
        default=300
    )
    parser.add_argument(
        '--P_value',
        type=int,
        default=12
    )

    FLAGS, _ = parser.parse_known_args()

    #main()
    remix(cal_path=None, val_path='./min_max_uint8_remixed_300_validation_P12')