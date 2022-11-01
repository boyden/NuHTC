import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as maskUtils

def mask_nms(masks, pred_scores, thr=0.9):
    """https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/common/maskApi.c#L98

    Returns:
        tuple: kept dets and indice.
    """
    sort_idx = np.argsort(pred_scores)[::-1]
    mask_len = len(masks)
    tmp_masks = np.array(masks)[sort_idx]
    mask_iou = maskUtils.iou(tmp_masks.tolist(), tmp_masks.tolist(), [0] * mask_len)

    keep_idx = np.ones(mask_len, dtype=np.uint8)
    for i in range(mask_len):
        if not keep_idx[i]:
            continue
        for j in range(i + 1, mask_len):
            if not keep_idx[j]:
                continue
            tmp_iou = mask_iou[i, j]
            # tmp_iou = maskUtils.iou([tmp_masks[i]], [tmp_masks[j]], [0])[0][0]
            if tmp_iou > thr:
                keep_idx[j] = 0
    return tmp_masks[keep_idx==1].tolist(), keep_idx
# --------------------------Optimised for Speed
def get_fast_aji(true_masks, pred_masks, pairwise_inter=None, pairwise_union=None):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from
    over-penalisation similar to DICE2.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.

    """
    # due to the overlap between some nucleis, there is a slight difference
    # compared to assign instance number first. Details of codes could be accessed from original hover net
    if pairwise_inter is None or pairwise_union is None:
        pairwise_inter, pairwise_union = get_pairwise_iou(true_masks, pred_masks)
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    true_id_list = range(len(true_masks))
    pred_id_list = range(len(pred_masks))
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true)  # index to instance ID
    paired_pred = list(paired_pred)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    return aji_score


#####
def get_fast_aji_plus(true_masks, pred_masks, pairwise_inter=None, pairwise_union=None, paired_true=None, paired_pred=None):
    """AJI+, an AJI version with maximal unique pairing to obtain overall intersecion.
    Every prediction instance is paired with at most 1 GT instance (1 to 1) mapping, unlike AJI
    where a prediction instance can be paired against many GT instances (1 to many).
    Remaining unpaired GT and Prediction instances will be added to the overall union.
    The 1 to 1 mapping prevents AJI's over-penalisation from happening.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.

    """
    if pairwise_inter is None or pairwise_union is None:
        pairwise_inter, pairwise_union = get_pairwise_iou(true_masks, pred_masks)
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)

    true_id_list = range(len(true_masks))
    pred_id_list = range(len(pred_masks))
    #### Munkres pairing to find maximal unique pairing
    if paired_true is None or paired_pred is None:
        paired_true, paired_pred = linear_sum_assignmesnt(-pairwise_iou)
    ### extract the paired cost and remove invalid pair
    paired_iou = pairwise_iou[paired_true, paired_pred]
    # now select all those paired with iou != 0.0 i.e have intersection
    paired_true = paired_true[paired_iou > 0.0]
    paired_pred = paired_pred[paired_iou > 0.0]
    paired_inter = pairwise_inter[paired_true, paired_pred]
    paired_union = pairwise_union[paired_true, paired_pred]
    paired_true = list(paired_true)  # index to instance ID
    paired_pred = list(paired_pred)
    overall_inter = paired_inter.sum()
    overall_union = paired_union.sum()
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    return aji_score

#####
def get_fast_pq(true_masks, pred_masks, pairwise_inter=None, pairwise_union=None, paired_true=None, paired_pred=None, match_iou=0.5):
    """`match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing. 

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.    
    
    Fast computation requires instance IDs are in contiguous orderding 
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand 
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                      pairing information to perform measurement
                    
    """
    assert match_iou >= 0.0, "Cant' be negative"
    if pairwise_inter is None or pairwise_union is None:
        pairwise_inter, pairwise_union = get_pairwise_iou(true_masks, pred_masks)
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)

    true_id_list = range(len(true_masks))
    pred_id_list = range(len(pred_masks))

    #
    # if match_iou >= 0.5:
    #     paired_iou = pairwise_iou[pairwise_iou > match_iou]
    #     pairwise_iou[pairwise_iou <= match_iou] = 0.0
    #     paired_true, paired_pred = np.nonzero(pairwise_iou)
    #     paired_iou = pairwise_iou[paired_true, paired_pred]
    #     paired_true += 1  # index is instance id - 1
    #     paired_pred += 1  # hence return back to original
    # else:
    # * Exhaustive maximal unique pairing
    #### Munkres pairing with scipy library
    # the algorithm return (row indices, matched column indices)
    # if there is multiple same cost in a row, index of first occurence
    # is return, thus the unique pairing is ensure
    # inverse pair to get high IoU as minimum
    if paired_true is None or paired_pred is None:
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
    ### extract the paired cost and remove invalid pair
    paired_iou = pairwise_iou[paired_true, paired_pred]

    # now select those above threshold level
    # paired with iou = 0.0 i.e no intersection => FP or FN
    paired_true = list(paired_true[paired_iou > match_iou])
    paired_pred = list(paired_pred[paired_iou > match_iou])
    paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list if idx not in paired_pred]

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


#####
def get_fast_dice(true_masks, pred_masks, pairwise_inter=None, pairwise_union=None, paired_true=None, paired_pred=None):
    """Ensemble dice."""
    if pairwise_inter is None or pairwise_union is None:
        pairwise_inter, pairwise_union = get_pairwise_iou(true_masks, pred_masks)
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    if paired_true is None or paired_pred is None:
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)

    true_id_list = np.arange(len(true_masks))
    pred_id_list = np.arange(len(pred_masks))

    overall_total = 0
    overall_inter = 0

    min_iou = 1e-4
    allowable = pairwise_iou[paired_true, paired_pred] >= min_iou
    paired_true = paired_true[allowable]
    paired_pred = paired_pred[allowable]

    # find indices of unmatched
    def _find_unmatched(id_li, matched):
        return id_li[~np.in1d(id_li, matched)]
    unmatched1 = _find_unmatched(true_id_list, paired_true)
    unmatched2 = _find_unmatched(pred_id_list, paired_pred)

    if len(paired_true) + len(paired_pred) == 0:
        return 1
    elif len(paired_true) * len(paired_true) == 0:
        return 0
    else:
        for idx in range(len(paired_true)):
            t_mask = true_masks[paired_true[idx]]
            p_mask = pred_masks[paired_pred[idx]]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            overall_total += total
            overall_inter += inter
        return 2 * overall_inter / overall_total

#####--------------------------As pseudocode
def get_dice_1(true, pred):
    """Traditional dice."""
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred

    return 2.0 * np.sum(inter) / np.sum(denom)


####
def get_dice_2(true, pred):
    """Ensemble Dice as used in Computational Precision Medicine Challenge."""
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))
    # remove background aka id 0
    true_id.remove(0)
    pred_id.remove(0)

    total_markup = 0
    total_intersect = 0
    for t in true_id:
        t_mask = np.array(true == t, np.uint8)
        for p in pred_id:
            p_mask = np.array(pred == p, np.uint8)
            intersect = p_mask * t_mask
            if intersect.sum() > 0:
                total_intersect += intersect.sum()
                total_markup += t_mask.sum() + p_mask.sum()
    if total_markup == 0 and total_intersect == 0:
        return 0
    return 2 * total_intersect / total_markup


#####
def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


#####
def pair_coordinates(setA, setB, radius):
    """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal 
    unique pairing (largest possible match) when pairing points in set B 
    against points in set A, using distance as cost function.

    Args:
        setA, setB: np.array (float32) of size Nx2 contains the of XY coordinate
                    of N different points 
        radius: valid area around a point in setA to consider 
                a given coordinate in setB a candidate for match
    Return:
        pairing: pairing is an array of indices
        where point at index pairing[0] in set A paired with point
        in set B at index pairing[1]
        unparedA, unpairedB: remaining poitn in set A and set B unpaired

    """
    # * Euclidean distance as the cost matrix
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric='euclidean')

    # * Munkres pairing with scipy library
    # the algorithm return (row indices, matched column indices)
    # if there is multiple same cost in a row, index of first occurence 
    # is return, thus the unique pairing is ensured
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

    # extract the paired cost and remove instances 
    # outside of designated radius
    pair_cost = pair_distance[indicesA, paired_indicesB]

    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]

    pairing = np.concatenate([pairedA[:,None], pairedB[:,None]], axis=-1)
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
    return pairing, unpairedA, unpairedB

def np_vec_no_jit_iou(bboxes1, bboxes2):
    """Fast, vectorized IoU.

    Source: https://medium.com/@venuktan/vectorized-intersection-over-union ...
            -iou-in-numpy-and-tensor-flow-4fa16231b63d

    Parameters
    -----------
    bboxes1 : np array
        columns encode bounding box corners xmin, ymin, xmax, ymax
    bboxes2 : np array
        same as bboxes 1

    Returns
    --------
    np array
        IoU values for each pair from bboxes1 & bboxes2

    """
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def map_bboxes_using_hungarian_algorithm(bboxes1, bboxes2, min_iou=1e-4):
    """Map bounding boxes using hungarian algorithm.

    Adapted from Lee A.D. Cooper.

    Parameters
    ----------
    bboxes1 : numpy array
        columns correspond to xmin, ymin, xmax, ymax

    bboxes2 : numpy array
        columns correspond to xmin, ymin, xmax, ymax

    min_iou : float
        minumum iou to match two bboxes to match to each other

    Returns
    -------
    np.array
        matched indices relative to x1, y1

    np.array
        matched indices relative to x2, y2, correspond to the first output

    np.array
        unmatched indices relative to x1, y1

    np.array
        unmatched indices relative to x2, y2

    """
    # generate cost matrix for mapping cells from user to anchors
    max_cost = 1 - min_iou
    costs = 1 - np_vec_no_jit_iou(bboxes1=bboxes1, bboxes2=bboxes2)
    costs[costs > max_cost] = 99.

    # perform hungarian algorithm mapping
    source, target = linear_sum_assignment(costs)

    # discard mappings that are non-allowable
    allowable = costs[source, target] <= max_cost
    source = source[allowable]
    target = target[allowable]

    # find indices of unmatched
    def _find_unmatched(coords, matched):
        potential = np.arange(coords.shape[0])
        return potential[~np.in1d(potential, matched)]
    unmatched1 = _find_unmatched(bboxes1, source)
    unmatched2 = _find_unmatched(bboxes2, target)

    return source, target, unmatched1, unmatched2

def get_pairwise_iou(true_masks, pred_masks):
    # prefill with value
    true_id_list = range(len(true_masks))
    pred_id_list = range(len(pred_masks))
    pairwise_inter = np.zeros([len(true_masks), len(pred_masks)], dtype=np.float64)
    pairwise_union = np.zeros([len(true_masks), len(pred_masks)], dtype=np.float64)

    # caching pairwise
    for true_idx in true_id_list:  # 0-th is background
        t_mask = true_masks[true_idx]
        for pred_idx in pred_id_list:
            p_mask = pred_masks[pred_idx]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_idx, pred_idx] = inter
            pairwise_union[true_idx, pred_idx] = total - inter
    #
    return pairwise_inter, pairwise_union

def stat_calc(true_masks, pred_masks, match_iou=0.5):

    if len(true_masks)+len(pred_masks) == 0:
        return {
                    'aji':1,
                    'aji_plus':1,
                    'dq':1,
                    'sq':1,
                    'pq':1,
                    'dice':1
                }
    if len(true_masks)*len(pred_masks) == 0:
        return {
                    'aji':0,
                    'aji_plus':0,
                    'dq':0,
                    'sq':0,
                    'pq':0,
                    'dice':0
                }
    # prefill with value
    pairwise_inter = np.zeros([len(true_masks), len(pred_masks)], dtype=np.float64)
    pairwise_union = np.zeros([len(true_masks), len(pred_masks)], dtype=np.float64)

    # caching pairwise
    for true_idx in range(len(true_masks)):  # 0-th is background
        t_mask = true_masks[true_idx]
        for pred_idx in range(len(pred_masks)):
            p_mask = pred_masks[pred_idx]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_idx, pred_idx] = inter
            pairwise_union[true_idx, pred_idx] = total - inter

    aji = get_fast_aji(true_masks, pred_masks, pairwise_inter=pairwise_inter, pairwise_union=pairwise_union)
    aji_plus = get_fast_aji_plus(true_masks, pred_masks, pairwise_inter=pairwise_inter, pairwise_union=pairwise_union)
    pq = get_fast_pq(true_masks, pred_masks, pairwise_inter=pairwise_inter, pairwise_union=pairwise_union, match_iou=match_iou)
    tp = len(pq[1][0])
    fp = len(pq[1][3])
    fn = len(pq[1][2])
    dice = get_fast_dice(true_masks, pred_masks, pairwise_inter=pairwise_inter, pairwise_union=pairwise_union)
    stat_res = {
        'aji':aji,
        'aji_plus':aji_plus,
        'dq':pq[0][0],
        'sq':pq[0][1],
        'pq':pq[0][2],
        'dice':dice,
        'precision':tp/(tp+fp+1e-9),
        'recall':tp/(tp+fn+1e-9),
    }
    return stat_res