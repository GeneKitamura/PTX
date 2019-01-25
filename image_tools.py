import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def _draw_bbox(img, x, y, w, h, label, color):

    cp_img = np.copy(img) # for cv2.addWeighted method to keep boxes transparent
    x, y, w, h = int(x), int(y), int(w), int(h)
    xmin = x
    xmax = x + w
    ymin = y
    ymax = y + h
    cv2.rectangle(cp_img, (xmin, ymin), (xmax, ymax), color, 2) # bounding box
    cv2.rectangle(cp_img, (xmin-1, ymin), (xmax+1, ymin-20), color, cv2.FILLED) # label box
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(cp_img, label, (xmin+5, ymin-5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    alpha = 0.8
    cv2.addWeighted(cp_img, alpha, img, 1.0 - alpha, 0, img) # last arg is destination

def draw_image_with_box(bbox_df, index_num, label='PTX', color=[255, 0, 0], pic_folder='./../CXR/images/'):
    img_loc, x, y, w, h = bbox_df.loc[index_num, ['Image Index', 'x', 'y', 'w', 'h']]
    img = cv2.imread(pic_folder + img_loc)
    org_h, org_w, _ = img.shape

    # proportional to absolute size
    x, w = x*org_w, w*org_w
    y, h = y*org_h, h*org_h

    f, ax = plt.subplots(figsize=(15, 15))

    _draw_bbox(img, x, y, w, h, label, color)

    ax.imshow(img)
    plt.show()

def resize_label_array(label_array, size_tuple):
    old_h = label_array.shape[0]
    old_w = label_array.shape[1]
    old_c = label_array.shape[2]

    new_h = size_tuple[0]
    new_w = size_tuple[1]
    resized_array = np.zeros((new_h, new_w, old_c))

    # With math.ceil, the boxes are going to be shifted down and to the right.  Consider a shape that is firmly divisible between the new and old shapes (300 by 20 is split evenly to 15; 300 by 12 is split to 25).  Or consider alternating between ceil and floor.
    h_ratio = math.ceil(new_h / old_h)
    w_ratio = math.ceil(new_w / old_w)

    for channel in range(old_c):
        for i in range(old_h):
            curr_row = i * h_ratio

            for j in range(old_w):
                curr_column = j * w_ratio
                resized_array[curr_row: (curr_row + h_ratio), curr_column : (curr_column + w_ratio), channel] = label_array[i, j, channel]

    return resized_array

def tile_alt_imshow(img_arrays, labels=None, titles=None, label_choice=1, width=30, height=30, save_it=None, h_slot=None, w_slot=None):
    scaled_img_arrays = rescale_img(img_arrays)
    #scaled_img_arrays = img_arrays
    img_n, img_h, img_w, _ = img_arrays.shape

    if h_slot is None:
        h_slot = int(math.ceil(np.sqrt(img_n)))
    if w_slot is None:
        w_slot = int(math.ceil(np.sqrt(img_n)))

    fig, axes = plt.subplots(h_slot, w_slot, figsize=(width, height))
    fig.subplots_adjust(hspace=0, wspace=0)

    for ax, i in zip(axes.flatten(), range(img_n)):
        img = rescale_img(scaled_img_arrays[i])
        #img = scaled_img_arrays[i]

        if labels is not None:
            c_labels = resize_label_array(labels[i], (img_h, img_w))
            img *= np.expand_dims(c_labels[..., label_choice], axis=2)
        ax.imshow(img)
        if titles is not None:
            ax.set_title(titles[i], color='red')

        ax.axis('off')

    if save_it is not None:
        plt.savefig(save_it, dpi=300, format='png')
    else:
        plt.show()

def alt_imshow(img, rescale=True, labels=None, label_choice=1):
    # Label choice to isolate finding of choice.

    if rescale:
        img = rescale_img(img)

    if labels is not None:
        # labels will black out an image
        labels = resize_label_array(labels, (img.shape[0], img.shape[1]))
        # labels = np.expand_dims(labels, 2)
        # labels = (labels == label_choice).astype(np.float32)
        img *= np.expand_dims(labels[..., label_choice], axis=2)

    f, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, 'gray')
    ax.axis('off')
    plt.show()

def rescale_img(img, new_min=0, new_max=1):
    # not perfect since old min and old max based on curr img, and not on whole dataset
    # works WELL when used on array of images
    return (img - np.min(img)) * (new_max - new_min) / (np.max(img) - np.min(img)) + new_min