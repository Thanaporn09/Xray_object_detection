_base_ = './faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt_MD',
        depth=101,
        num_stages=4,
	groups=32,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
	mdconv=dict(type='MDConv', use_deform=False),
        stage_with_mdconv=(False, True, True, True),
        style='pytorch'),
    neck=dict(
        type='Fusion_FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        type='StandardRoIHead_point',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead_point',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=5,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            point_coder=dict(
                type='DeltaXYWHBBoxCoder_point',
                target_means=[0., 0., 0., 0., 0., 0., 0.1, 0.1, 0.1, 0.1],
                target_stds=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            loss_point=dict(type='L1Loss', loss_weight=1.0))))
