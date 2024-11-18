"""run.

Usage:
  compute_stats.py --true_path=<n> --pred_path=<n> --type_path=<n> --save_path=<n> [--num_classes=<n>]
  compute_stats.py (-h | --help)
  compute_stats.py --version

Options:
  -h --help             Show this string.
  --version             Show version.
  --true_path=<n>    Root path to where the ground-truth is saved.
  --pred_path=<n>    Root path to where the predictions are saved.
  --type_path=<n>    Root path to where the types are saved.
  --save_path=<n>    Path where the prediction CSV files will be saved
  --num_classes=<n>   The number of the classes. [default: 5].
"""


import docopt
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('/home/bal753/NuHTC/tools/analysis_tools/pannuke')
from utils import get_fast_pq, remap_label, binarize, get_coord_pq
import tqdm
tissue_types = [
                'Adrenal_gland',
                'Bile-duct',
                'Bladder',
                'Breast',
                'Cervix',
                'Colon',
                'Esophagus',
                'HeadNeck',
                'Kidney',
                'Liver',
                'Lung',
                'Ovarian',
                'Pancreatic',
                'Prostate',
                'Skin',
                'Stomach',
                'Testis',
                'Thyroid',
                'Uterus'
                ]
def format_metric_det(mPQ_all, bPQ_all, save_path, types, metric='TP'):
    metric = metric
    # mPQ_each_image = [np.sum(pq) for pq in mPQ_all]
    # bPQ_each_image = [np.sum(pq_bin) for pq_bin in bPQ_all]

    # class metric
    total_PQ = np.sum(bPQ_all)
    neo_PQ = np.sum([pq[0] for pq in mPQ_all])
    inflam_PQ = np.sum([pq[1] for pq in mPQ_all])
    conn_PQ = np.sum([pq[2] for pq in mPQ_all])
    dead_PQ = np.sum([pq[3] for pq in mPQ_all])
    nonneo_PQ = np.sum([pq[4] for pq in mPQ_all])

    # Print for each class
    print('Printing calculated metrics on a single split')
    print('-'*40)
    print(f'Total {metric}: {total_PQ}')
    print(f'Neoplastic {metric}: {neo_PQ}')
    print(f'Inflammatory {metric}: {inflam_PQ}')
    print(f'Connective {metric}: {conn_PQ}')
    print(f'Dead {metric}: {dead_PQ}')
    print(f'Non-Neoplastic {metric}: {nonneo_PQ}')
    print('-' * 40)

    # Save per-class metrics as a csv file
    for_dataframe = {'Class Name': ['Neoplastic', 'Inflam', 'Connective', 'Dead', 'Non-Neoplastic', 'Total'],
                        metric: [neo_PQ, inflam_PQ, conn_PQ, dead_PQ, nonneo_PQ, total_PQ]}
    df = pd.DataFrame(for_dataframe, columns=['Class Name', metric])
    df.to_csv(save_path + f'/class_stats_{metric}.csv')

def format_metric(mPQ_all, bPQ_all, save_path, types, metric='PQ'):
    metric = metric
    mPQ_each_image = [np.nanmean(pq) for pq in mPQ_all]
    bPQ_each_image = [np.nanmean(pq_bin) for pq_bin in bPQ_all]

    # class metric
    total_PQ = np.nanmean(bPQ_all)
    neo_PQ = np.nanmean([pq[0] for pq in mPQ_all])
    inflam_PQ = np.nanmean([pq[1] for pq in mPQ_all])
    conn_PQ = np.nanmean([pq[2] for pq in mPQ_all])
    dead_PQ = np.nanmean([pq[3] for pq in mPQ_all])
    nonneo_PQ = np.nanmean([pq[4] for pq in mPQ_all])

    # Print for each class
    print('Printing calculated metrics on a single split')
    print('-'*40)
    print(f'Neoplastic {metric}: {neo_PQ}')
    print(f'Inflammatory {metric}: {inflam_PQ}')
    print(f'Connective {metric}: {conn_PQ}')
    print(f'Dead {metric}: {dead_PQ}')
    print(f'Non-Neoplastic {metric}: {nonneo_PQ}')
    print('-' * 40)

    # Save per-class metrics as a csv file
    for_dataframe = {'Class Name': ['Neoplastic', 'Inflam', 'Connective', 'Dead', 'Non-Neoplastic', 'Total'],
                        metric: [neo_PQ, inflam_PQ, conn_PQ, dead_PQ, nonneo_PQ, total_PQ]}
    df = pd.DataFrame(for_dataframe, columns=['Class Name', metric])
    df.to_csv(save_path + f'/class_stats_{metric}.csv')

    # Print for each tissue
    all_tissue_mPQ = []
    all_tissue_bPQ = []
    for tissue_name in tissue_types:
        indices = [i for i, x in enumerate(types) if x == tissue_name]
        tissue_PQ = [mPQ_each_image[i] for i in indices]
        print(f'{tissue_name} {metric}: {np.nanmean(tissue_PQ)} ')
        tissue_PQ_bin = [bPQ_each_image[i] for i in indices]
        print(f'{tissue_name} {metric} binary: {np.nanmean(tissue_PQ_bin)} ')
        all_tissue_mPQ.append(np.nanmean(tissue_PQ))
        all_tissue_bPQ.append(np.nanmean(tissue_PQ_bin))

    # Save per-tissue metrics as a csv file
    for_dataframe = {'Tissue name': tissue_types + ['mean'],
                        metric: all_tissue_mPQ + [np.nanmean(all_tissue_mPQ)] , f'{metric} bin': all_tissue_bPQ + [np.nanmean(all_tissue_bPQ)]}
    df = pd.DataFrame(for_dataframe, columns=['Tissue name', metric, f'{metric} bin'])
    df.to_csv(save_path + f'/tissue_stats_{metric}.csv')

    # Show overall metrics - mPQ is average PQ over the classes and the tissues, bPQ is average binary PQ over the tissues
    print('-' * 40)
    print(f'Average m{metric}:{np.nanmean(all_tissue_mPQ)}')
    print(f'Average b{metric}:{np.nanmean(all_tissue_bPQ)}')

def main_coord(args):
    """
    This function returns the statistics reported on the PanNuke dataset, reported in the paper below:

    Gamper, Jevgenij, Navid Alemi Koohbanani, Simon Graham, Mostafa Jahanifar, Syed Ali Khurram,
    Ayesha Azam, Katherine Hewitt, and Nasir Rajpoot.
    "PanNuke Dataset Extension, Insights and Baselines." arXiv preprint arXiv:2003.10778 (2020).

    Args:
    Root path to the ground-truth
    Root path to the predictions
    Path where results will be saved

    Output:
    Terminal output of bPQ and mPQ results for each class and across tissues
    Saved CSV files for bPQ and mPQ results for each class and across tissues
    """

    true_root = args['--true_path']
    pred_root = args['--pred_path']
    save_path = args['--save_path']
    type_path = args['--type_path']
    num_classes = int(args['--num_classes'])

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if os.path.splitext(true_root)[1] != '':
        true_path = true_root
        true_root = os.path.dirname(true_root)
    else:
        true_path = os.path.join(true_root,'masks.npy')  # path to the GT for a specific split

    if os.path.splitext(pred_root)[1] != '':
        pred_path = pred_root
    else:
        pred_path = os.path.join(pred_root, 'masks.npy')  # path to the predictions for a specific split

    if os.path.splitext(type_path)[1] == '':
        type_path = os.path.join(type_path,'types.npy') # path to the nuclei types

    # load the data
    true = np.load(true_path)
    pred = np.load(pred_path)
    types = np.load(type_path)

    mPQ_all = []
    bPQ_all = []
    mDQ_all = []
    bDQ_all = []
    mPrecision_all = []
    bPrecision_all = []
    mRecall_all = []
    bRecall_all = []
    mTP_all = []
    bTP_all = []
    mFP_all = []
    bFP_all = []
    mFN_all = []
    bFN_all = []
    # loop over the images
    for i in tqdm.tqdm(range(true.shape[0])):
        pq, dq, precision, recall, tp_li, fp_li, fn_li = [], [], [], [], [], [], []
        pred_bin = binarize(pred[i,:,:,:num_classes])
        pred_bin = remap_label(pred_bin)
        true_bin = binarize(true[i,:,:,:num_classes])

        if len(np.unique(true_bin)) == 1:
            dq_bin = np.nan
            precision_bin = np.nan
            recall_bin = np.nan
            tp_bin = 0
            fp_bin = 0
            fn_bin = 0
        else:
            [dq_bin, tp, fn, fp] = get_coord_pq(true_bin, pred_bin) # compute PQ
            precision_bin = len(tp) / (len(tp) + len(fp) + 1e-9)
            recall_bin = len(tp) / (len(tp) + len(fn) + 1e-9)
            tp_bin = len(tp)
            fp_bin = len(fp)
            fn_bin = len(fn)


        # loop over the classes
        for j in range(num_classes):
            pred_tmp = pred[i,:,:,j]
            pred_tmp = pred_tmp.astype('int32')
            true_tmp = true[i,:,:,j]
            true_tmp = true_tmp.astype('int32')
            pred_tmp = remap_label(pred_tmp)
            true_tmp = remap_label(true_tmp)

            if len(np.unique(true_tmp)) == 1:
                dq_tmp = np.nan # if ground truth is empty for that class, skip from calculation
                precision_tmp = np.nan
                recall_tmp = np.nan
            else:
                [dq_tmp, tp_tmp, fn_tmp, fp_tmp] = get_coord_pq(true_tmp, pred_tmp) # compute PQ
                precision_tmp = len(tp_tmp) / (len(tp_tmp) + len(fp_tmp) + 1e-9)
                recall_tmp = len(tp_tmp) / (len(tp_tmp) + len(fn_tmp) + 1e-9)
            dq.append(dq_tmp)
            precision.append(precision_tmp)
            recall.append(recall_tmp)
            tp_li.append(len(tp_tmp))
            fp_li.append(len(fp_tmp))
            fn_li.append(len(fn_tmp))

        mDQ_all.append(dq)
        bDQ_all.append([dq_bin])
        mPrecision_all.append(precision)
        bPrecision_all.append([precision_bin])
        mRecall_all.append(recall)
        bRecall_all.append([recall_bin])
        mTP_all.append(tp_li)
        bTP_all.append([tp_bin])
        mFP_all.append(fp_li)
        bFP_all.append([fp_bin])
        mFN_all.append(fn_li)
        bFN_all.append([fn_bin])
    # using np.nanmean skips values with nan from the mean calculation
    # format_metric(mPQ_all, bPQ_all, save_path, types, metric='PQ')
    format_metric(mDQ_all, bDQ_all, save_path, types, metric='DQ')
    format_metric(mPrecision_all, bPrecision_all, save_path, types, metric='Precision')
    format_metric(mRecall_all, bRecall_all, save_path, types, metric='Recall')
    format_metric_det(mTP_all, bTP_all, save_path, types, metric='TP')
    format_metric_det(mFP_all, bFP_all, save_path, types, metric='FP')
    format_metric_det(mFN_all, bFN_all, save_path, types, metric='FN')

def main_iou(args):
    """
    This function returns the statistics reported on the PanNuke dataset, reported in the paper below:

    Gamper, Jevgenij, Navid Alemi Koohbanani, Simon Graham, Mostafa Jahanifar, Syed Ali Khurram,
    Ayesha Azam, Katherine Hewitt, and Nasir Rajpoot.
    "PanNuke Dataset Extension, Insights and Baselines." arXiv preprint arXiv:2003.10778 (2020).

    Args:
    Root path to the ground-truth
    Root path to the predictions
    Path where results will be saved

    Output:
    Terminal output of bPQ and mPQ results for each class and across tissues
    Saved CSV files for bPQ and mPQ results for each class and across tissues
    """

    true_root = args['--true_path']
    pred_root = args['--pred_path']
    save_path = args['--save_path']
    type_path = args['--type_path']
    num_classes = int(args['--num_classes'])

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if os.path.splitext(true_root)[1] != '':
        true_path = true_root
        true_root = os.path.dirname(true_root)
    else:
        true_path = os.path.join(true_root,'masks.npy')  # path to the GT for a specific split

    if os.path.splitext(pred_root)[1] != '':
        pred_path = pred_root
    else:
        pred_path = os.path.join(pred_root, 'masks.npy')  # path to the predictions for a specific split

    if os.path.splitext(type_path)[1] == '':
        type_path = os.path.join(type_path,'types.npy') # path to the nuclei types

    # load the data
    true = np.load(true_path)
    pred = np.load(pred_path)
    types = np.load(type_path)

    mPQ_all = []
    bPQ_all = []
    mDQ_all = []
    bDQ_all = []
    mPrecision_all = []
    bPrecision_all = []
    mRecall_all = []
    bRecall_all = []
    mTP_all = []
    bTP_all = []
    mFP_all = []
    bFP_all = []
    mFN_all = []
    bFN_all = []
    # loop over the images
    for i in tqdm.tqdm(range(true.shape[0])):
        pq, dq, precision, recall, tp_li, fp_li, fn_li = [], [], [], [], [], [], []
        pred_bin = binarize(pred[i,:,:,:num_classes])
        pred_bin = remap_label(pred_bin)
        true_bin = binarize(true[i,:,:,:num_classes])

        if len(np.unique(true_bin)) == 1:
            pq_bin = np.nan # if ground truth is empty for that class, skip from calculation
            dq_bin = np.nan
            precision_bin = np.nan
            recall_bin = np.nan
            tp_bin = 0
            fp_bin = 0
            fn_bin = 0
        else:
            [dq_bin, _, pq_bin], [tp, _, fn, fp] = get_fast_pq(true_bin, pred_bin) # compute PQ
            precision_bin = len(tp) / (len(tp) + len(fp) + 1e-9)
            recall_bin = len(tp) / (len(tp) + len(fn) + 1e-9)
            tp_bin = len(tp)
            fp_bin = len(fp)
            fn_bin = len(fn)


        # loop over the classes
        for j in range(num_classes):
            pred_tmp = pred[i,:,:,j]
            pred_tmp = pred_tmp.astype('int32')
            true_tmp = true[i,:,:,j]
            true_tmp = true_tmp.astype('int32')
            pred_tmp = remap_label(pred_tmp)
            true_tmp = remap_label(true_tmp)

            if len(np.unique(true_tmp)) == 1:
                pq_tmp, dq_tmp = np.nan, np.nan # if ground truth is empty for that class, skip from calculation
            else:
                [dq_tmp, _, pq_tmp] , [tp_tmp, _, fn_tmp, fp_tmp] = get_fast_pq(true_tmp, pred_tmp) # compute PQ
                precision_tmp = len(tp_tmp) / (len(tp_tmp) + len(fp_tmp) + 1e-9)
                recall_tmp = len(tp_tmp) / (len(tp_tmp) + len(fn_tmp) + 1e-9)
            pq.append(pq_tmp)
            dq.append(dq_tmp)
            precision.append(precision_tmp)
            recall.append(recall_tmp)
            tp_li.append(len(tp_tmp))
            fp_li.append(len(fp_tmp))
            fn_li.append(len(fn_tmp))

        mPQ_all.append(pq)
        bPQ_all.append([pq_bin])
        mDQ_all.append(dq)
        bDQ_all.append([dq_bin])
        mPrecision_all.append(precision)
        bPrecision_all.append([precision_bin])
        mRecall_all.append(recall)
        bRecall_all.append([recall_bin])
        mTP_all.append(tp_li)
        bTP_all.append([tp_bin])
        mFP_all.append(fp_li)
        bFP_all.append([fp_bin])
        mFN_all.append(fn_li)
        bFN_all.append([fn_bin])
    # using np.nanmean skips values with nan from the mean calculation
    format_metric(mPQ_all, bPQ_all, save_path, types, metric='PQ')
    format_metric(mDQ_all, bDQ_all, save_path, types, metric='DQ')
    format_metric(mPrecision_all, bPrecision_all, save_path, types, metric='Precision')
    format_metric(mRecall_all, bRecall_all, save_path, types, metric='Recall')
    format_metric_det(mTP_all, bTP_all, save_path, types, metric='TP')
    format_metric_det(mFP_all, bFP_all, save_path, types, metric='FP')
    format_metric_det(mFN_all, bFN_all, save_path, types, metric='FN')

#####
if __name__ == '__main__':
    args = docopt.docopt(__doc__, version='PanNuke Evaluation v1.0')
    main_coord(args)

