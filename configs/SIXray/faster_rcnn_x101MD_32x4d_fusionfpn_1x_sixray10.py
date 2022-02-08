_base_ = './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
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
	bbox_head=dict(
		num_classes=5
	)))
