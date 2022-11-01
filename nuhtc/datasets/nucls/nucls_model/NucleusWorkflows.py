import sys
import os

from os.path import join as opj
import matplotlib.pylab as plt
import numpy as np
from pandas import Series, DataFrame, read_csv, concat
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from shutil import copyfile
from histomicstk.annotations_and_masks.annotations_to_masks_handler import \
    _visualize_annotations_on_rgb
from histomicstk.annotations_and_masks.masks_to_annotations_handler import \
    get_contours_from_mask
from histomicstk.features.compute_nuclei_features import \
    compute_nuclei_features
from histomicstk.preprocessing.color_deconvolution import \
    color_deconvolution_routine

from ..GeneralUtils import maybe_mkdir
from .DataLoadingUtils import NucleusDataset, \
    get_cv_fold_slides, _crop_all_to_fov, NucleusDatasetMask, \
    NucleusDatasetMask_IMPRECISE
import PlottingUtils as pu
from .FasterRCNN import FasterRCNN
from .PartialMaskRCNN import PartialMaskRCNN
from .ModelRunner import trainNucleusModel, load_ckp, \
    evaluateNucleusModel
from ..configs.nucleus_model_configs import CoreSetQC, CoreSetNoQC, \
    EvalSets, VisConfigs
from .MiscUtils import map_bboxes_using_hungarian_algorithm
from .DataFormattingUtils import parse_sparse_mask_for_use
from .MaskRCNN import MaskRCNN
from ..configs.nucleus_style_defaults import NucleusCategories as ncg
from pprint import pprint

# %%===========================================================================

# noinspection DuplicatedCode
def run_one_fasterrcnn_fold(
        fold: int, cfg, model_root: str, model_name: str, train=True,
        vis_test=True):

    # FIXME: for prototyping
    if fold == 999:
        cfg.FasterRCNNConfigs.training_params.update({
            'effective_batch_size': 4,
            'smoothing_window': 1,
            'test_evaluate_freq': 1,
        })

    model_folder = opj(model_root, f'fold_{fold}')
    maybe_mkdir(model_folder)
    checkpoint_path = opj(model_folder, f'{model_name}.ckpt')

    # %% --------------------------------------------------------------
    # Init model

    model = FasterRCNN(**cfg.FasterRCNNConfigs.fastercnn_params)

    # %% --------------------------------------------------------------
    # Test that it works in forward mode

    # model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    # predictions = model(x)

    # %% --------------------------------------------------------------
    # Prep data loaders

    train_slides, test_slides = get_cv_fold_slides(
        train_test_splits_path=CoreSetQC.train_test_splits_path, fold=fold)

    # copy train/test slides with model itself just to be safe
    for tr in ('train', 'test'):
        fname = f'fold_{fold}_{tr}.csv'
        copyfile(
            opj(CoreSetQC.train_test_splits_path, fname),
            opj(model_folder, fname),
        )

    train_dataset = NucleusDataset(
        root=CoreSetQC.dataset_root, dbpath=CoreSetQC.dbpath,
        slides=train_slides, **cfg.BaseDatasetConfigs.train_dataset)

    test_dataset = NucleusDataset(
        root=CoreSetQC.dataset_root, dbpath=CoreSetQC.dbpath,
        slides=test_slides, **cfg.BaseDatasetConfigs.test_dataset)

    # handle class imbalance
    if cfg.FasterRCNNConfigs.handle_class_imbalance:
        del cfg.BaseDatasetConfigs.train_loader['shuffle']
        cfg.BaseDatasetConfigs.train_loader['sampler'] = WeightedRandomSampler(
            weights=train_dataset.fov_weights,
            num_samples=len(train_dataset.fov_weights),
            replacement=cfg.FasterRCNNConfigs.sample_with_replacement,
        )

    # %% --------------------------------------------------------------
    # Train model

    if train:
        trainNucleusModel(
            model=model, checkpoint_path=checkpoint_path,
            data_loader=DataLoader(
                dataset=train_dataset, **cfg.BaseDatasetConfigs.train_loader),
            data_loader_test=DataLoader(
                dataset=test_dataset, **cfg.BaseDatasetConfigs.test_loader),
            **cfg.FasterRCNNConfigs.training_params)

    elif os.path.exists(checkpoint_path):
        ckpt = load_ckp(checkpoint_path=checkpoint_path, model=model)
        model = ckpt['model']

    # %% --------------------------------------------------------------
    # Visualize some predictions

    n_predict = 15
    min_iou = 0.5

    maybe_mkdir(opj(model_folder, 'predictions'))

    if vis_test:
        dataset = test_dataset
    else:
        dataset = train_dataset

    # cropper = tvdt.Cropper()

    model.eval()
    model.to('cpu')

    for imno in range(n_predict):

        # pick one image from the dataset
        imgtensor, target = dataset.__getitem__(imno)
        imname = dataset.rfovids[int(target['image_id'])]

        print(f"predicting image {imno} of {n_predict}: {imname}")

        # get prediction
        with torch.no_grad():
            output = model([imgtensor.to('cpu')])
        cpu_device = torch.device("cpu")
        output = [{k: v.to(cpu_device) for k, v in t.items()} for t in output]
        output = output[0]

        # # crop the prediction to FOV
        # imgtensor, target, output = _crop_all_to_fov(
        #     images=[imgtensor], targets=[target], outputs=[output],
        #     cropper=cropper)
        # imgtensor = imgtensor[0]
        # target = target[0]
        # output = output[0]

        # Ignore ambiguous nuclei from matching. Note that the
        #  model already filters out anything predicted as ignore_label
        #  in inference mode, so we only need to do this for gtruth
        keep = target['iscrowd'] == 0
        trg_boxes = np.int32(target['boxes'][keep])

        # get true/false positives/negatives
        output_boxes = np.int32(output['boxes'])
        _, TP, FN, FP = map_bboxes_using_hungarian_algorithm(
            bboxes1=trg_boxes, bboxes2=output_boxes, min_iou=min_iou)

        # concat relevant bounding boxes
        relevant_bboxes = np.concatenate((
            output_boxes[TP], output_boxes[FP], trg_boxes[FN],
        ), axis=0)
        match_colors = [VisConfigs.MATCHING_COLORS['TP']] * len(TP) \
            + [VisConfigs.MATCHING_COLORS['FP']] * len(FP) \
            + [VisConfigs.MATCHING_COLORS['FN']] * len(FN)

        # get rgb
        rgb = np.uint8(imgtensor * 255.).transpose(1, 2, 0)

        # visualize true bounding boxes
        nperrow = 3
        nrows = 1
        fig, ax = plt.subplots(
            nrows, nperrow, figsize=(5 * nperrow, 5.3 * nrows))

        # truth
        axis = ax[0]
        axis.imshow(rgb)
        axis.set_title('rgb', fontsize=12)

        # prediction (objectness)
        axis = ax[1]
        axis = pu.vis_bbox(
            img=rgb, bbox=relevant_bboxes, ax=axis,
            instance_colors=match_colors, linewidth=1.5,
        )
        axis.set_title('bboxes (TP/FP/FN)', fontsize=12)

        # visualize prediction (classification)
        axis = ax[2]
        output_colors = Series(
            np.int32(output['labels'])).map(dataset.rlabelcodes).map(
            VisConfigs.CATEG_COLORS)
        axis = pu.vis_bbox(
            img=rgb, bbox=output_boxes, ax=axis,
            instance_colors=output_colors.tolist(),
            linewidth=1.5,
        )
        axis.set_title('prediction (classif.)', fontsize=12)

        # plt.show()
        plt.savefig(opj(model_folder, f'predictions/{imno}_{imname}.png'))
        plt.close()

# %%===========================================================================


# noinspection DuplicatedCode
def run_one_maskrcnn_fold(
        fold: int, cfg, model_root: str, model_name: str, qcd_training=True,
        train=True, vis_test=True, n_vis=100, randomvis=True):

    # FIXME: for prototyping
    if fold == 999:
        cfg.MaskRCNNConfigs.training_params.update({
            'effective_batch_size': 4,
            'smoothing_window': 1,
            'test_evaluate_freq': 1,
        })

    model_folder = opj(model_root, f'fold_{fold}')
    maybe_mkdir(model_folder)
    checkpoint_path = opj(model_folder, f'{model_name}.ckpt')

    # %% --------------------------------------------------------------
    # Init model
    print('\nInit model')
    pprint(cfg.MaskRCNNConfigs.maskrcnn_params)
    model = MaskRCNN(**cfg.MaskRCNNConfigs.maskrcnn_params)

    # %% --------------------------------------------------------------
    # Test that it works in forward mode

    # model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    # predictions = model(x)

    # %% --------------------------------------------------------------
    # Prep data loaders
    print('\nPrep data loaders')
    train_slides, test_slides = get_cv_fold_slides(
        train_test_splits_path=CoreSetQC.train_test_splits_path, fold=fold)

    # copy train/test slides with model itself just to be safe
    for tr in ('train', 'test'):
        fname = f'fold_{fold}_{tr}.csv'
        copyfile(
            opj(CoreSetQC.train_test_splits_path, fname),
            opj(model_folder, fname),
        )

    # training data optionally QCd
    if qcd_training:
        print('\ntraining data optionally QCd')
        pprint(cfg.MaskDatasetConfigs.train_dataset)
        train_dataset = NucleusDatasetMask(
            root=CoreSetQC.dataset_root, dbpath=CoreSetQC.dbpath,
            slides=train_slides, **cfg.MaskDatasetConfigs.train_dataset)
    else:
        print('\ntraining data')
        pprint(cfg.MaskDatasetConfigs.train_dataset)
        train_dataset = NucleusDatasetMask(
            root=CoreSetNoQC.dataset_root, dbpath=CoreSetNoQC.dbpath,
            slides=train_slides, **cfg.MaskDatasetConfigs.train_dataset)

    # test set is always the QC'd data
    test_dataset = NucleusDatasetMask(
        root=CoreSetQC.dataset_root, dbpath=CoreSetQC.dbpath,
        slides=test_slides, **cfg.MaskDatasetConfigs.test_dataset)

    # handle class imbalance
    if cfg.MaskRCNNConfigs.handle_class_imbalance:
        del cfg.BaseDatasetConfigs.train_loader['shuffle']
        cfg.BaseDatasetConfigs.train_loader['sampler'] = WeightedRandomSampler(
            weights=train_dataset.fov_weights,
            num_samples=len(train_dataset.fov_weights),
            replacement=cfg.MaskRCNNConfigs.sample_with_replacement,
        )

    # %% --------------------------------------------------------------
    # Train model

    if train:
        print('\nTrain model')
        pprint(cfg.MaskDatasetConfigs.train_loader)
        trainNucleusModel(
            model=model, checkpoint_path=checkpoint_path,
            data_loader=DataLoader(
                dataset=train_dataset, **cfg.MaskDatasetConfigs.train_loader),
            data_loader_test=DataLoader(
                dataset=test_dataset, **cfg.MaskDatasetConfigs.test_loader),
            **cfg.MaskRCNNConfigs.training_params)

    elif os.path.exists(checkpoint_path):
        ckpt = load_ckp(checkpoint_path=checkpoint_path, model=model)
        model = ckpt['model']

    # %% --------------------------------------------------------------
    # Visualize some predictions

    min_iou = 0.5
    vis_props = {'linewidth': 0.15, 'text': False}

    maybe_mkdir(opj(model_folder, 'predictions'))

    if vis_test:
        dataset = test_dataset
    else:
        dataset = train_dataset

    # cropper = tvdt.Cropper()

    model.eval()
    model.to('cpu')

    if randomvis:
        tovis = list(np.random.choice(len(dataset), size=(n_vis,)))
    else:
        tovis = list(range(n_vis))

    for imidx, imno in enumerate(tovis):

        # pick one image from the dataset
        imgtensor, target = dataset.__getitem__(imno)
        imname = dataset.rfovids[int(target['image_id'])]

        print(f"predicting image {imidx} of {n_vis}: {imname}")

        # get prediction
        with torch.no_grad():
            output = model([imgtensor.to('cpu')])
        cpu_device = torch.device('cpu')
        output = [{k: v.to(cpu_device) for k, v in t.items()} for t in output]
        output = output[0]

        # mTODO?: the cropper does not support sparse masks
        # # crop the prediction to FOV

        # Ignore ambiguous nuclei from matching. Note that the
        #  model already filters out anything predicted as ignore_label
        #  in inference mode, so we only need to do this for gtruth
        keep = target['iscrowd'] == 0
        trg_boxes = np.int32(target['boxes'][keep])

        # get true/false positives/negatives
        output_boxes = np.int32(output['boxes'])
        _, TP, FN, FP = map_bboxes_using_hungarian_algorithm(
            bboxes1=trg_boxes, bboxes2=output_boxes, min_iou=min_iou)

        # concat relevant bounding boxes
        relevant_bboxes = np.concatenate((
            output_boxes[TP], output_boxes[FP], trg_boxes[FN],
        ), axis=0)
        match_colors = [VisConfigs.MATCHING_COLORS['TP']] * len(TP) \
            + [VisConfigs.MATCHING_COLORS['FP']] * len(FP) \
            + [VisConfigs.MATCHING_COLORS['FN']] * len(FN)

        # just to comply with histomicstk default style
        rgtcodes = {
            k: {
                'group': v,
                'color': f'rgb(' + ','.join(str(c) for c in VisConfigs.CATEG_COLORS[v]) + ')',
            }
            for k, v in dataset.rlabelcodes.items()
        }

        # extract contours +/ condensed masks (truth)
        # noinspection PyTupleAssignmentBalance
        _, _, contoursdf_truth = parse_sparse_mask_for_use(
            sparse_mask=np.uint8(target['masks']),
            rgtcodes=rgtcodes, labels=target['labels'].tolist(),
        )

        # extract contours +/ condensed masks (prediction)
        output_labels = np.int32(output['labels'])
        output_labels = output_labels.tolist()
        if not model.transform.densify_mask:
            # output mask is sparse
            # noinspection PyTupleAssignmentBalance
            _, _, contoursdf_prediction = parse_sparse_mask_for_use(
                sparse_mask=np.uint8(output['masks'][:, 0, :, :] > 0.5),
                rgtcodes=rgtcodes, labels=output_labels,
            )
        else:
            # output mask is already dense
            contoursdf_prediction = get_contours_from_mask(
                MASK=output['masks'].numpy(),
                GTCodes_df=DataFrame.from_records(data=[
                    {
                        'group': rgtcodes[label]['group'],
                        'GT_code': idx + 1,
                        'color': rgtcodes[label]['color']
                    }
                    for idx, label in enumerate(output_labels)
                ]),
                MIN_SIZE=1,
                get_roi_contour=False,
            )

        # get rgb
        rgb = np.uint8(imgtensor * 255.).transpose(1, 2, 0)

        # visualize bounding boxes and masks
        nperrow = 4
        nrows = 1
        fig, ax = plt.subplots(nrows, nperrow,
                               figsize=(5 * nperrow, 5.3 * nrows))

        # just the image
        axis = ax[0]
        axis.imshow(rgb)
        axis.set_title('rgb', fontsize=12)

        # relevant predicted (TP, FP) & true (FN) boxes
        axis = ax[1]
        axis = pu.vis_bbox(
            img=rgb, bbox=relevant_bboxes, ax=axis,
            instance_colors=match_colors, linewidth=1.5,
        )
        axis.set_title('Bboxes detection (TP/FP/FN)', fontsize=12)

        # predicted masks
        axis = ax[2]
        prediction_vis = _visualize_annotations_on_rgb(
            rgb=rgb,
            contours_list=contoursdf_prediction.to_dict(orient='records'),
            **vis_props)
        axis.imshow(prediction_vis)
        axis.set_title('Predicted masks + classif.', fontsize=12)

        # true masks
        axis = ax[3]
        truth_vis = _visualize_annotations_on_rgb(
            rgb=rgb, contours_list=contoursdf_truth.to_dict(orient='records'),
            **vis_props)
        axis.imshow(truth_vis)
        axis.set_title('True masks/bboxes + classif.', fontsize=12)

        # plt.show()
        plt.savefig(opj(model_folder, f'predictions/{imno}_{imname}.png'))
        plt.close()

# %%===========================================================================


# noinspection DuplicatedCode
def evaluate_maskrcnn_fold_on_inferred_truth(
        fold: int, cfg, model_root: str, model_name: str,
        whoistruth='Ps', evalset='E', getmetrics=True, n_vis=100):

    model_folder = opj(model_root, f'fold_{fold}')
    checkpoint_path = opj(model_folder, f'{model_name}.ckpt')
    savepath = opj(model_folder, f'Eval_{whoistruth}AreTruth_{evalset}')
    maybe_mkdir(savepath)

    # %% --------------------------------------------------------------
    # Init model

    model = MaskRCNN(**cfg.MaskRCNNConfigs.maskrcnn_params)

    # %% --------------------------------------------------------------
    # Prep data loaders

    slides = read_csv(opj(
        model_folder, f'fold_{fold}_test.csv')).loc[:, 'slide_name'].tolist()
    dataset = NucleusDatasetMask(
        root=EvalSets.dataset_roots[evalset][whoistruth],
        dbpath=EvalSets.dbpaths[evalset][whoistruth],
        slides=slides, **cfg.MaskDatasetConfigs.test_dataset)

    # %% --------------------------------------------------------------
    # Evaluate model

    ckpt = load_ckp(checkpoint_path=checkpoint_path, model=model)
    model = ckpt['model']

    if getmetrics:
        ecfgs = {
            k: v for k, v in cfg.MaskRCNNConfigs.training_params.items() if k in [
                'test_maxDets',
                'n_testtime_augmentations',
                'crop_inference_to_fov'
            ]
        }
        tsls = evaluateNucleusModel(
            model=model, checkpoint_path=checkpoint_path,
            dloader=DataLoader(
                dataset=dataset, **cfg.MaskDatasetConfigs.test_loader),
            **ecfgs)

        # save results
        for i, tsl in enumerate(tsls):
            with open(opj(savepath, f'testingMetrics_{i}.txt'), 'w') as f:
                f.write(str(tsl)[1:-1].replace(', ', '\n'))

    # %% --------------------------------------------------------------
    # Visualize some predictions

    min_iou = 0.5
    vis_props = {'linewidth': 0.15, 'text': False}

    maybe_mkdir(opj(savepath, 'predictions'))

    # cropper = tvdt.Cropper()

    model.eval()
    model.to('cpu')

    for imno in range(min(n_vis, len(dataset))):

        # pick one image from the dataset
        imgtensor, target = dataset.__getitem__(imno)
        imname = dataset.rfovids[int(target['image_id'])]

        print(f"visualizing image {imno} of {n_vis}: {imname}")

        # get prediction
        with torch.no_grad():
            output = model([imgtensor.to('cpu')])
        cpu_device = torch.device('cpu')
        output = [{k: v.to(cpu_device) for k, v in t.items()} for t in output]
        output = output[0]

        # mTODO?: the cropper does not support sparse masks
        # # crop the prediction to FOV

        # Ignore ambiguous nuclei from matching. Note that the
        #  model already filters out anything predicted as ignore_label
        #  in inference mode, so we only need to do this for gtruth
        keep = target['iscrowd'] == 0
        trg_boxes = np.int32(target['boxes'][keep])

        # get true/false positives/negatives
        output_boxes = np.int32(output['boxes'])
        _, TP, FN, FP = map_bboxes_using_hungarian_algorithm(
            bboxes1=trg_boxes, bboxes2=output_boxes, min_iou=min_iou)

        # concat relevant bounding boxes
        relevant_bboxes = np.concatenate((
            output_boxes[TP], output_boxes[FP], trg_boxes[FN],
        ), axis=0)
        match_colors = [VisConfigs.MATCHING_COLORS['TP']] * len(TP) \
            + [VisConfigs.MATCHING_COLORS['FP']] * len(FP) \
            + [VisConfigs.MATCHING_COLORS['FN']] * len(FN)

        # just to comply with histomicstk default style
        rgtcodes = {
            k: {
                'group': v,
                'color': f'rgb(' + ','.join(str(c) for c in VisConfigs.CATEG_COLORS[v]) + ')',
            }
            for k, v in dataset.rlabelcodes.items()
        }

        # extract contours +/ condensed masks (truth)
        # noinspection PyTupleAssignmentBalance
        dense_mask, _, contoursdf_truth = parse_sparse_mask_for_use(
            sparse_mask=np.uint8(target['masks']),
            rgtcodes=rgtcodes, labels=target['labels'].tolist(),
        )

        # extract contours +/ condensed masks (prediction)
        output_labels = np.int32(output['labels'])
        output_labels = output_labels.tolist()
        if not model.transform.densify_mask:
            # output mask is sparse
            # noinspection PyTupleAssignmentBalance
            _, _, contoursdf_prediction = parse_sparse_mask_for_use(
                sparse_mask=np.uint8(output['masks'][:, 0, :, :] > 0.5),
                rgtcodes=rgtcodes, labels=output_labels,
            )
        else:
            # output mask is already dense
            contoursdf_prediction = get_contours_from_mask(
                MASK=output['masks'].numpy(),
                GTCodes_df=DataFrame.from_records(data=[
                    {
                        'group': rgtcodes[label]['group'],
                        'GT_code': idx + 1,
                        'color': rgtcodes[label]['color']
                    }
                    for idx, label in enumerate(output_labels)
                ]),
                MIN_SIZE=1,
                get_roi_contour=False,
            )

        # get rgb
        rgb = np.uint8(imgtensor * 255.).transpose(1, 2, 0)

        # visualize bounding boxes and masks
        nperrow = 4
        nrows = 1
        fig, ax = plt.subplots(nrows, nperrow,
                               figsize=(5 * nperrow, 5.3 * nrows))

        # just the image
        axis = ax[0]
        axis.imshow(rgb)
        axis.set_title('rgb', fontsize=12)

        # relevant predicted (TP, FP) & true (FN) boxes
        axis = ax[1]
        axis = pu.vis_bbox(
            img=rgb, bbox=relevant_bboxes, ax=axis,
            instance_colors=match_colors, linewidth=1.5,
        )
        axis.set_title('Bboxes detection (TP/FP/FN)', fontsize=12)

        # predicted masks
        axis = ax[2]
        prediction_vis = _visualize_annotations_on_rgb(
            rgb=rgb,
            contours_list=contoursdf_prediction.to_dict(orient='records'),
            **vis_props)
        axis.imshow(prediction_vis)
        axis.set_title('Predicted masks + classif.', fontsize=12)

        # true masks
        axis = ax[3]
        truth_vis = _visualize_annotations_on_rgb(
            rgb=rgb, contours_list=contoursdf_truth.to_dict(orient='records'),
            **vis_props)
        axis.imshow(truth_vis)
        axis.set_title('True masks/bboxes + classif.', fontsize=12)

        # plt.show()
        plt.savefig(opj(savepath, f'predictions/{imno}_{imname}.png'))
        plt.close()

# %%===========================================================================


def _parse_nucleus_metas_df(nuclidxs, dataset, target, output):

    n_nuclei = len(nuclidxs)

    nmetas = DataFrame(index=nuclidxs)
    fovname = dataset.rfovids[int(target['image_id'])]
    nmetas.loc[:, 'fovname'] = fovname
    nmetas.loc[:, 'slide_name'] = fovname.split('_')[0]
    nmetas.loc[:, 'hospital'] = fovname.split('-')[1]
    boxes = DataFrame(
        output['boxes'].numpy(), index=nuclidxs,
        columns=['xmin', 'ymin', 'xmax', 'ymax'])
    objectness = DataFrame(
        output['scores'].numpy(), index=nuclidxs, columns=['objectness'])
    categs = DataFrame(
        Series(output['labels'].numpy(), index=nuclidxs).map(
            dataset.rlabelcodes),
        columns=['pred_categ'])
    categs_probabs = DataFrame(
        output['probabs'].numpy(), index=nuclidxs,
        columns=[f'pred_probab_{c}' for c in dataset.categs_names])

    # matched gtruth
    matched_tidxs, matched_didxs, _, _ = \
        map_bboxes_using_hungarian_algorithm(
            bboxes1=np.int32(target['boxes']),
            bboxes2=np.int32(boxes.values),
            min_iou=0.5,
        )
    tcategs = DataFrame(index=nuclidxs)
    tcategs.loc[:, 'ismatched'] = 0
    relidxs = np.in1d(np.arange(n_nuclei), matched_didxs)
    tcategs.loc[relidxs, 'ismatched'] = 1
    tlabs = np.int32(target['labels'])[matched_tidxs]
    tcs = list(map(lambda x: dataset.rlabelcodes[x], tlabs))
    tcategs.loc[relidxs, 'true_categ'] = tcs

    # now concat
    nmetas = concat([
        nmetas, boxes, objectness,
        categs, categs_probabs, tcategs,
    ], axis=1)

    return nmetas


# noinspection DuplicatedCode
@torch.no_grad()
def get_maskrcnn_representations(
        fold: int, cfg, model_root: str, model_name: str,
        savedir: str, subset='train'):

    assert subset in ['train', 'test']

    # paths
    model_folder = opj(model_root, f'fold_{fold}')
    checkpoint_path = opj(model_folder, f'{model_name}.ckpt')
    p1 = opj(savedir, model_name)
    savepath = opj(p1, f'fold_{fold}_{subset}_FEATS')
    maybe_mkdir(p1)
    maybe_mkdir(savepath)

    # load model
    model = PartialMaskRCNN(**cfg.MaskRCNNConfigs.maskrcnn_params)
    ckpt = load_ckp(checkpoint_path=checkpoint_path, model=model)
    model = ckpt['model']
    model.eval()
    cpu_device = torch.device('cpu')
    model.transform.densify_mask = False

    # prep data loader
    slides = read_csv(
        opj(model_folder, f'fold_{fold}_{subset}.csv')
    ).loc[:, 'slide_name'].tolist()
    dataset = NucleusDatasetMask(
        root=CoreSetQC.dataset_root, dbpath=CoreSetQC.dbpath,
        slides=slides, **cfg.MaskDatasetConfigs.test_dataset,
    )
    dataiter = iter(dataset)

    NUCLID = 0

    # go through fovs and fetch features
    nfovs = len(dataset)
    for fovno in range(nfovs):

        print(f'fov {fovno + 1} of {nfovs}')
        overwrite = NUCLID < 1

        # do inference
        imgtensor, target = next(dataiter)
        output = model([imgtensor])
        output = [{k: v.to(cpu_device) for k, v in t.items()} for t in output]
        output = output[0]
        n_nuclei = len(output['labels'])
        nuclidxs = [f'nucls_{i + NUCLID}' for i in range(1, n_nuclei + 1)]

        # metadata about each detection
        nmetas = _parse_nucleus_metas_df(
            nuclidxs=nuclidxs, dataset=dataset, target=target, output=output)
        nmetas.to_csv(
            opj(savepath, f'nucleus_metadata.csv'),
            mode='w' if overwrite else 'a', header=overwrite)

        # save box features (-> box regression)
        #    & cboxfeatures & class_logits (-> classif)
        fdescs = {
            'box_features': 'bfeat', 'cbox_features': 'cbfeat',
            'clogits': 'clogits'}
        for ftype, fdesc in fdescs.items():
            bfeatures = DataFrame(output[ftype].numpy())
            bfeatures.columns = [f'{fdesc}_{i + 1}' for i in bfeatures.columns]
            bfeatures.index = nuclidxs
            bfeatures.to_csv(
                opj(savepath, f'{ftype}.csv'),
                mode='w' if overwrite else 'a', header=overwrite)

        # interpretation-friendly features
        rgb = np.uint8(imgtensor * 255.).transpose(1, 2, 0)
        stains, _, _ = color_deconvolution_routine(rgb)
        htx = 255 - stains[..., 0]
        masks = np.uint8(output['masks'].numpy() > 0.5)[:, 0, :, :]
        ifeatures = []
        for nid in range(n_nuclei):
            try:
                fdf = compute_nuclei_features(
                    im_label=masks[nid, ...], im_nuclei=htx)
                fdf.index = [nuclidxs[nid]]
            except:
                fdf = DataFrame(index=[nuclidxs[nid]])
            ifeatures.append(fdf)
        ifeatures = concat(ifeatures, axis=0)
        ifeatures.to_csv(
            opj(savepath, f'interp_features.csv'),
            mode='w' if overwrite else 'a', header=overwrite)

        # dont forget
        NUCLID += n_nuclei

# %%===========================================================================
