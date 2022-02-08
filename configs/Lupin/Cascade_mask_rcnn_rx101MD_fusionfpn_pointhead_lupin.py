_base_ = './configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet_MD',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
	mdconv=dict(type='MDConv', use_deform=False),
        stage_with_mdconv=(False, True, True, True),
        style='pytorch'),
    neck=dict(
        type='Fusion_FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        type='CascadeRoIHead_point',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead_point',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                point_coder=dict(
                    type='DeltaXYWHBBoxCoder_point',
                    target_means=[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0),
                loss_point=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead_point',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                point_coder=dict(
                    type='DeltaXYWHBBoxCoder_point',
                    target_means=[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0),
                loss_point=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead_point',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                        point_coder=dict(
                    type='DeltaXYWHBBoxCoder_point',
                    target_means=[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033, 0.033]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                loss_point=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0))
        ],
	mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=8,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))))
