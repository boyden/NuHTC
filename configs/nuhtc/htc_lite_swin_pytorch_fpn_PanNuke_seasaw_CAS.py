'''
Example:
    >>> from mmdet.models import ResNet
    >>> import torch
    >>> self = ResNet(depth=18)
    >>> self.eval()
    >>> inputs = torch.rand(1, 3, 32, 32)
    >>> level_outputs = self.forward(inputs)
    >>> for level_out in level_outputs:
    ...     print(tuple(level_out.shape))
    (1, 64, 8, 8)
    (1, 128, 4, 4)
    (1, 256, 2, 2)
    (1, 512, 1, 1)
'''
# CUDA_VISIBLE_DEVICES=1 nohup python tools/train.py configs/nuhtc/htc_lite_swin_pytorch_fpn_PanNuke_seasaw_CAS.py > WSI_Seg_PanNuke_HTC_lite_swin_fold1.log 2>&1 &
# ps aux | grep WSI_Seg_HTC_swin_PanNuke.py | awk '{print $2}' | xargs kill -9
fold = 1
thres = 0.965926
num_classes = 5
scale_factor = 2.0
max_epochs = 200
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],  std=[58.395, 57.12, 57.375], to_rgb=True)
dist_params = dict(backend='nccl')
log_level = 'INFO'
dataset_name = 'PanNuke'
dataset = f'{dataset_name}CocoDataset'
workflow = [('train', 1)]
mmdet_base = '../../thirdparty/mmdetection/configs/_base_'
fp16 = dict(loss_scale='dynamic')
neck_type = 'FPN'
log_note = f'{neck_type}_AttenROI_thres_{int(thres*100)}_base_aug_cas'
log_name = f'htc_lite_swin_pytorch_seasaw_{log_note}_{dataset_name}_full_epoch_{max_epochs}_fold{fold}'
work_dir = f'./work_dirs/{log_name}'
# PATH that need to be replaced
data_dir = './dataset'
basedir = f'{data_dir}/{dataset_name}'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
load_from = None
resume_from = None
by_epoch = True
seg_ignore_label = 0

model=dict(
    type='HybridTaskCascade_Cus',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type=neck_type,
        in_channels=[96, 192, 384, 768],
        out_channels=64,
        num_outs=4),
    rpn_head=dict(
        type='RPNHead',
        in_channels=64,
        feat_channels=64,
        reg_decoded_bbox=False,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', loss_weight=1.0, beta=1.0/9.0)),
    roi_head=dict(
        type='HybridTaskCascadeRoIHead_Lite',
        interleaved=True,
        mask_info_flow=True,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        watershed_proposal=True,
        bbox_roi_extractor=dict(
            type='AttentionRoIExtractor',
            start_level=2,
            thres=thres,
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=64,
            # finest_scale=35,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHeadWithProb',
                in_channels=64,
                fc_out_channels=256,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                cls_predictor_cfg=dict(type='NormedLinear', tempearture=20),
                loss_cls=dict(
                    type='SeesawLoss',
                    p=0.8,
                    q=2.0,
                    num_classes=num_classes,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHeadWithProb',
                in_channels=64,
                fc_out_channels=256,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                cls_predictor_cfg=dict(type='NormedLinear', tempearture=20),
                loss_cls=dict(
                    type='SeesawLoss',
                    p=0.8,
                    q=2.0,
                    num_classes=num_classes,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHeadWithProb',
                in_channels=64,
                fc_out_channels=256,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                cls_predictor_cfg=dict(type='NormedLinear', tempearture=20),
                loss_cls=dict(
                    type='SeesawLoss',
                    p=0.8,
                    q=2.0,
                    num_classes=num_classes,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        ],
        mask_roi_extractor=dict(
            type='AttentionRoIExtractor',
            start_level=2,
            thres=thres,
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=64,
            # finest_scale=35,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=[
            dict(
                type='HTCMaskHead',
                with_conv_res=True,
                num_convs=4,
                in_channels=64,
                conv_out_channels=64,
                class_agnostic=True,
                num_classes=num_classes,
                loss_mask=dict(type='DiceLoss', loss_weight=1.0)),
        ],
        semantic_roi_extractor=dict(
            type='AttentionRoIExtractor',
            start_level=2,
            thres=thres,
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=64,
            # finest_scale=35,
            featmap_strides=[4]),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=4,
            fusion_level=0,
            num_convs=4,
            in_channels=64,
            conv_out_channels=64,
            num_classes=1,
            loss_seg=dict(type='DiceLoss', loss_weight=1.0)),
    ),
    train_cfg=dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=1024,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_pre=6000,
        max_per_img=3000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=10),
    rcnn=[
        dict(
            assigner=dict(
                type='MaskIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=1024,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaskIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=1024,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaskIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=1024,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)
    ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=10),
        rcnn=dict(
            score_thr=0.35,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300,
            mask_thr_binary=0.5))
)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    # dict(type='RandomCrop', crop_size=(256, 256)),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=9),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandCorrupt', prob=0.5),
    dict(
        type='Sequential',
        transforms=[
            dict(
                type='OneOf',
                transforms=[{
                            'type': 'RandTranslate',
                            'x': (-0.1, 0.1),
                            'seg_ignore_label': seg_ignore_label
                        }, {
                            'type': 'RandTranslate',
                            'y': (-0.1, 0.1),
                            'seg_ignore_label': seg_ignore_label
                        }, {
                            'type': 'RandTranslate',
                            'x': (-0.1, 0.1),
                            'y': (-0.1, 0.1),
                            'seg_ignore_label': seg_ignore_label
                        }, {
                            'type': 'RandRotate',
                            'angle': (-10, 10),
                            'seg_ignore_label': seg_ignore_label
                        },[{
                                'type': 'RandShear',
                                'x': (-10, 10),
                                'seg_ignore_label': seg_ignore_label
                            }, {
                                'type': 'RandShear',
                                'y': (-10, 10),
                                'seg_ignore_label': seg_ignore_label
                            }]
                        ]),
            dict(type='RecomputeBox'),
        ],
        record=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 4),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=scale_factor,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        # _delete_=True,
        type='CASDataset',
        dataset=dict(
            type='PanNukeCocoDataset',
            ann_file=f'./coco/PanNuke/PanNuke_annt_RLE_fold{(fold-1)%3+1}.json',
            img_prefix=f'{basedir}/rgb/',
            seg_prefix=f'{basedir}/rgb_seg',
            pipeline=train_pipeline,
        )
    ),
    val=dict(
        type='PanNukeCocoDataset',
        ann_file=f'./coco/PanNuke/PanNuke_annt_RLE_fold{(fold+4)%3+1}.json',
        img_prefix=f'{basedir}/rgb/',
        seg_prefix=f'{basedir}/rgb_seg',
        pipeline=test_pipeline,
    ),
    test=dict(
        type='PanNukeCocoDataset',
        ann_file=f'./coco/PanNuke/PanNuke_annt_RLE_fold{(fold+4)%3+1}.json',
        img_prefix=f'{basedir}/rgb/',
        seg_prefix=f'{basedir}/rgb_seg',
        pipeline=test_pipeline,
    ),
    )


evaluation = dict(interval=10, metric=['bbox', 'segm', 'proposal'], by_epoch=by_epoch)
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[30, 160])
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=10, by_epoch=by_epoch, max_keep_ckpts=40)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook_Cus',
            commit=True,
            init_kwargs=dict(
                project=dataset_name,
                name=log_name,
                config=dict(
                    work_dirs=work_dir,
                    total_step=max_epochs,
                )),
            by_epoch=by_epoch)
    ])
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='WeightSummary'),
    dict(type='Mask_Vis_Hook', interval=2000),
    dict(type='LinearMomentumEMAHook', momentum=0.0002, warm_up=100, priority=40),
    dict(type='FineTune', iter=15000),
]

