import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import metrics
from collections import namedtuple, defaultdict
from scipy import interpolate

Data = namedtuple('Data', ['images', 'labels', 'nus', 'singles'])
Data_indexed = namedtuple('Data', ['images', 'labels', 'nus', 'singles', 'pd_indices'])
truncated_Data = namedtuple('truncated_Data', ['images', 'singles'])

def path_dict():

    path2id = {'Atelectasis': 9,
 'Cardiomegaly': 2,
 'Consolidation': 14,
 'Edema': 13,
 'Effusion': 4,
 'Emphysema': 3,
 'Fibrosis': 12,
 'Hernia': 5,
 'Infiltration': 6,
 'Mass': 7,
 'No Finding': 0,
 'Nodule': 8,
 'Pleural_Thickening': 10,
 'Pneumonia': 11,
 'Pneumothorax': 1
           }

    id2path = {i: j for j, i in path2id.items()}

    return path2id, id2path

def eval_metrics(predictions, label_hot, num_classes=2, plot_auc=False, plot_precision_recall=False, print_short_sum=True,
                 print_synopsis=True, return_values=False, color='black', save_auc=False, return_roc_values=False):
    fpr = {}
    tpr = {}
    precision = {}
    recall = {}
    auc = {}
    roc_thresholds = {}
    avg_precision = {}
    # label_hot is one hot whether class is {0, 1} denoted as an array of size [n, 2]

    for i in range(num_classes):
        # predictions is a dictionary
        class_prediction = predictions[i]
        cur_labels = label_hot[..., i]
        fpr[i], tpr[i], roc_thresholds[i] = metrics.roc_curve(y_true=cur_labels, y_score=class_prediction)
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_true=cur_labels, probas_pred=class_prediction)
        avg_precision[i] = metrics.average_precision_score(y_true=cur_labels, y_score=class_prediction)
        auc[i] = metrics.auc(fpr[i], tpr[i])

    #tot_predictions = np.concatenate([np.expand_dims(j, 1) for i, j in predictions.items()], axis=1)
    tot_predictions = np.stack([predictions[0], predictions[1]], axis=1)

    # micro handles all cases together
    #fp['micro'], tp['micro'], thresholds['micro'] = metrics.roc_curve(y_true=label_hot.ravel(), y_score=tot_predictions.ravel())
    #auc['micro'] = metrics.auc(fp['micro'], tp['micro'])

    ind_prediction = np.argmax(tot_predictions, axis=1)
    ind_true = np.argmax(label_hot, axis=1)

    # prec is PPV for class 1 and NPV for class 0
    # recall is sensitivity for class 1 and specificity for class 0
    prec, rec, f1, sup = metrics.precision_recall_fscore_support(y_true=ind_true, y_pred=ind_prediction)
    avg_p, avg_r, avg_f1, _ = metrics.precision_recall_fscore_support(y_true=ind_true, y_pred=ind_prediction, average='weighted')
    synopsis = metrics.classification_report(y_true=ind_true, y_pred=ind_prediction)
    accuracy = metrics.accuracy_score(y_true=ind_true, y_pred=ind_prediction)

    #pos_auc = metrics.roc_auc_score(y_true=one_hot_labels[..., 1], y_score=predictions[1])

    if plot_auc:
        f, ax = plt.subplots()
        #ax.plot(fpr[0], tpr[0], label='Class 0 AUC is %.2f' % auc[0], marker='X')
        ax.plot(fpr[1], tpr[1], label='Pneumothorax detection AUC is %.2f' % auc[1], marker='o', markevery=20)
        #ax.plot(fp['micro'], tp['micro'], label='Average AUC is %.2f' % auc['micro'], marker='v')
        ax.plot([0, 1], [0, 1], linestyle='dashed')
        ax.set_xlabel("False Positive rate", color=color)
        ax.set_ylabel("True Positive rate", color=color)
        ax.set_xlim(left=-0.01, right=1.01)
        ax.set_ylim(bottom=0, top=1.01)
        ax.tick_params(color=color, labelcolor=color)
        #ax.set_title('Receiver Operator Curve', color=color)
        ax.legend(loc=4)
        if save_auc:
            f.savefig('./AUC_curve_v3', dpi=600, format='png')
        plt.show()

    if plot_precision_recall:
        f, ax = plt.subplots()
        ax.step(recall[1], precision[1], label='Class 1 average precision is %.2f' % avg_precision[1], marker='o', where='post', markersize=0)
        #ax.plot(recall[1], precision[1], label='Class 1 average precision is %.2f' % avg_precision[1], marker='o', markevery=10)
        ax.set_xlabel("Recall", color=color)
        ax.set_ylabel("Precision", color=color)
        ax.set_xlim(left=0, right=1.00)
        ax.set_ylim(bottom=0, top=1.01)
        ax.tick_params(color=color, labelcolor=color)
        ax.set_title('Precision Recall Curve', color=color)
        ax.legend(loc=1)
        plt.show()

    if print_synopsis:
        print(synopsis)

    if print_short_sum:
        for i, item in auc.items():
            print('AUC for class {0} is {1:.2f}'.format(i, item))
        print('Accuracy is {0:.2f}:'.format(accuracy))

    if return_values:
        return recall, precision

    if return_roc_values:
        print('returning roc values')
        return fpr, tpr, roc_thresholds

def combined_roc_curves():
    pass

def whole_sample_jaccard(sam_labels, sam_pred, patch_threshold, IOU_threshold):
    total_n = sam_labels.shape[0]
    above_T = 0

    for i in range(total_n):
        IOU_T = jaccard_score(sam_labels[i], sam_pred[i], patch_threshold, IOU_threshold)
        above_T += IOU_T

    return above_T / total_n

def jaccard_score(true_labels, pred_labels, patch_threshold, IOU_threshold, eps=1e-7, c_class=1):
    c_pred_labels = np.float32(pred_labels[..., c_class] > patch_threshold)
    summed_boxes = true_labels[..., c_class] + c_pred_labels #Intersection results in 2, non-I Union results in 1, negatives stay 0
    uniques, counts = np.unique(summed_boxes, return_counts=True)
    count_dict = defaultdict(lambda: 0, dict(zip(uniques, counts)))

    c_value = (count_dict[2] / (count_dict[2] + count_dict[1] + eps))
    return c_value > IOU_threshold


def rescale(array, new_min, new_max, old_min, old_max):
    return (array - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

def selective_initializer(sess, graph):
    with graph.as_default():
        global_var = tf.global_variables()

    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_var])
    not_initialized = [v for v, f in zip(global_var, is_not_initialized) if not f]
    sess.run(tf.variables_initializer(not_initialized))

def safe_sigmoid(mdarr):

    def _sig_pos(i_mdarr):
        return 1 / (1 + tf.exp(-i_mdarr))
    def _sig_neg(i_mdarr):
        return tf.exp(i_mdarr) / (1 + tf.exp(i_mdarr))

    pos_logits = mdarr >= 0
    neg_logits = tf.logical_not(pos_logits)
    pos_nums = mdarr * tf.cast(pos_logits, tf.float32)
    neg_nums = mdarr * tf.cast(neg_logits, tf.float32)
    pos_output = _sig_pos(pos_nums)
    neg_output = _sig_neg(neg_nums)

    return tf.where(pos_logits, pos_output, neg_output)

def exp_norm_softmax(d4array):
    class_max = tf.reduce_max(d4array, axis=3, keepdims=True)
    mod_inp = tf.exp(d4array - class_max)
    return mod_inp / tf.reduce_sum(mod_inp, axis=3, keepdims=True)

def create_holders(num_classes):
    pred_dict = {i: np.array([], np.float32) for i in range(num_classes)}
    labels_list = np.empty((0, num_classes), np.float32)
    return pred_dict, labels_list

def erase_borders(current_image):

    # checking the rows from the top
    for i in range (current_image.shape[0]):
        pixel_check = current_image[i][0]

        # Need to break out of loop when pixels are no longer homogeneous across row
        if np.mean(current_image[i] == pixel_check) != 1:
            current_image = current_image[i:]
            break

    # checking rows from the bottom
    for i in range(current_image.shape[0]-1, 0, -1): # subtract one at starting point to prevent IndexError
        pixel_check = current_image[i][0]

        if np.mean(current_image[i] == pixel_check) != 1:
            current_image = current_image[:i]
            break

    # checking columns from one side
    for i in range(current_image.shape[1]):
        pixel_check = current_image[0][i]

        # Need to break out of loop when pixels are not longer homogeneous through column
        if np.mean(current_image[:, i] == pixel_check) != 1:
            current_image = current_image[:, i:]
            break

    # checking columns from the other side
    for i in range(current_image.shape[1] - 1, 0, -1):
        pixel_check = current_image[0][i]

        if np.mean(current_image[:, i] == pixel_check) != 1:
            current_image = current_image[:, :i]
            break

    return current_image
