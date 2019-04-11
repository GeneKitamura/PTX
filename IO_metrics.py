import numpy as np
import sklearn
import matplotlib.pyplot as plt

from QC_helpers import print_shapes

def save_inferred_values(pred_dict, labels_list, collect_patches, filename):
    np.savez(filename, pred_0=pred_dict[0], pred_1=pred_dict[1], labels=labels_list, patches=collect_patches)

def load_inferred_values(filename):
    with np.load(filename) as f:
        pred_0 = f['pred_0']
        pred_1 = f['pred_1']
        labels = f['labels']
        patches = f['patches']

    preds = {}
    preds[0] = pred_0
    preds[1] = pred_1

    return preds, labels, patches

def roc_auc(predictions, labels, return_thresholds=True, plot_auc=True, color='black', save_auc=False):

    labels_argmax = np.argmax(labels, axis=1)

    fpr = {}
    tpr = {}
    roc_thresholds = {}
    auc = {}

    for i in [1]:
        fpr[i], tpr[i], roc_thresholds[i] = sklearn.metrics.roc_curve(y_true=labels_argmax, y_score=predictions[1])
        auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

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

    if return_thresholds:
        return roc_thresholds[1]

def combined_roc_curve(predict_list, labels_list, n_lines=3, color='black', save_auc=False):

    fpr = {}
    tpr = {}
    roc_thresholds = {}
    auc = {}

    for i in range(n_lines):
        labels_argmax = np.argmax(labels_list[i], axis=1)

        fpr[i], tpr[i], roc_thresholds[i] = sklearn.metrics.roc_curve(y_true=labels_argmax, y_score=predict_list[i][1])
        auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    f, ax = plt.subplots()
    ax.plot(fpr[0], tpr[0], label='CXR8 data on CXR8 model with AUC of %.2f' % auc[0], marker='X', markevery=20, linestyle=':')
    ax.plot(fpr[1], tpr[1], label='Our data on CXR8 model with AUC %.2f' % auc[1], marker='o', markevery=20, linestyle='-.')
    ax.plot(fpr[2], tpr[2], label='Our data on retrained model with AUC of %.2f' % auc[2], marker='^', markevery=20, linestyle='--')
    # ax.plot(fp['micro'], tp['micro'], label='Average AUC is %.2f' % auc['micro'], marker='v')
    ax.plot([0, 1], [0, 1], linestyle='-')
    ax.set_xlabel("False Positive rate", color=color)
    ax.set_ylabel("True Positive rate", color=color)
    ax.set_xlim(left=-0.01, right=1.01)
    ax.set_ylim(bottom=0, top=1.01)
    ax.tick_params(color=color, labelcolor=color)
    ax.set_title('Receiver Operator Curve', color=color)
    ax.legend(loc=4)
    if save_auc:
        f.savefig('./AUC_curve_combo.eps', dpi=400, format='eps')
    plt.show()



# Workflow is infer, then roc to get thresholds, then get metrics
def show_metrics(predict_1, labels, thresh_1, l_range=600, h_range=700, save_name=None):

    labels_argmax = np.argmax(labels, axis=1)
    eps = 1e-5

    thresh_list = []
    sens_list = []
    spec_list = []
    ppv_list = []
    npv_list = []
    acc_list = []

    best_comb = 0
    best_thresh = 0
    best_sens = 0
    best_spec = 0

    for i in range(l_range, h_range):
        a_thresholded = np.array(predict_1 > thresh_1[i], np.int)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(labels_argmax, a_thresholded).ravel()
        sens = tp / (tp + fn + eps)
        spec = tn / (tn + fp + eps)
        ppv = tp / (tp + fp + eps)
        npv = tn / (tn + fn + eps)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)

        comb_val = sens + spec
        if comb_val > best_comb:
            best_comb = comb_val
            best_thresh = i
            best_sens = sens
            best_spec = spec

        thresh_list.append(i)
        sens_list.append(sens)
        spec_list.append(spec)
        ppv_list.append(ppv)
        npv_list.append(npv)
        acc_list.append(acc)

        #print(i, sens, spec, ppv, npv, acc)

    print('Best threshold is: {} with sensitivity of {} and specificity of {}'.format(best_thresh, best_sens, best_spec))

    thresh_list = np.array(thresh_list)
    sens_list = np.array(sens_list)
    spec_list = np.array(spec_list)
    ppv_list = np.array(ppv_list)
    npv_list = np.array(npv_list)
    acc_list = np.array(acc_list)

    print_shapes([thresh_list, sens_list, spec_list, ppv_list, npv_list, acc_list])

    if save_name is not None:
        np.savez(save_name, threshold=thresh_list, sensitivity=sens_list, specificity=spec_list,
             ppv=ppv_list, npv=npv_list, accuracy=acc_list)

def load_metrics(filename, print_out=False):
    with np.load(filename) as f:
        thresh_list = f['threshold']
        sens_list = f['sensitivity']
        spec_list = f['specificity']
        ppv_list = f['ppv']
        npv_list = f['npv']
        acc_list = f['accuracy']

    print_shapes([thresh_list, sens_list, spec_list, ppv_list, npv_list, acc_list])

    if print_out:
        for i in range(thresh_list.shape[0]):
            print(thresh_list[i], sens_list[i], spec_list[i], ppv_list[i], npv_list[i], acc_list[i])


# Best values for TFR: thresh 659 with sensitivity of 0.7892790981648966 and specificity of 0.8288669057294362
# Best value for UP on TFR: threshold is: 40 with sensitivity of 0.33333326388890333 and specificity of 0.8500388434340417
# Best value for UP on UP: threshold is: 41 with sensitivity of 0.7708331727430889 and specificity of 0.8904428835241424