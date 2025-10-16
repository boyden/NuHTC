# --------------------------
# Called in calc_metrics.py for segmentation visualization (Last updated 09/09/2025)
# --------------------------

import os
import cv2
import sys
import random
import colorsys
import numpy as np
import scipy.io as sio
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
from matplotlib.ticker import MultipleLocator

def calculate_confusion_matrix(confusion_matrix, true_masks, pred_masks, gt_labels, pred_labels, tp_iou_thr=0.5):
    
    mask_ious = maskUtils.iou(true_masks, pred_masks, [0] * len(pred_masks))
    true_positives = np.zeros_like(gt_labels)
    for i, det_label in enumerate(pred_labels):
        det_match = 0
        for j, gt_label in enumerate(gt_labels):
            if mask_ious[j, i] >= tp_iou_thr:
                det_match += 1
                # if gt_label == det_label:
                true_positives[j] += 1  # TP
                confusion_matrix[gt_label, det_label] += 1
        if det_match == 0:  # BG FP
            confusion_matrix[-1, det_label] += 1
    for num_tp, gt_label in zip(true_positives, gt_labels):
        if num_tp == 0:  # FN
            confusion_matrix[gt_label, -1] += 1


def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_path=None,
                          show=False,
                          title='Normalized Confusion Matrix',
                          color_theme='plasma',
                          wandb_log=False):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_path (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `plasma`.
    """
    
    # Normalize
    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(num_classes, num_classes * 0.9), dpi=300)
    canvas = fig.canvas
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    title_font = {'weight': 'bold', 'size': 16}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 12}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # Draw confusion matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}%'.format(int(confusion_matrix[i, j])),
                ha='center',
                va='center',
                color='w',
                size=12)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    stream, (width, height) = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    cm_img = buffer.reshape((height, width, -1))
    if save_path is not None:
        plt.savefig(save_path, format='png')
        print(f'\nFigure saved to {save_path}.')
    if show:
        plt.show()
    plt.close()

    