


_base_ = [
    './_base_/retinanet_r50_fpn.py',
    './_base_/retinanet_tta.py'
]









conv_cfg = dict(type='ConvWS')  # weight standrized
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)  # group norm
mean=[132.54923396796545,63.94361466986094,30.328172035922037]
std=[20.02071647413696,11.869571542958491,6.3491549356145205]


# model settings
model = dict(
    type='FSAF',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=mean,
        std=std,
        bgr_to_rgb=True,
        pad_size_divisor=32),
     backbone=dict(
        type='ResNeXt',
        depth=50,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://jhu/resnext50_32x4d_gn_ws')
    ),
    bbox_head=dict(
        type='FSAFHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        reg_decoded_bbox=True,
        # Only anchor-free branch is implemented. The anchor generator only
        #  generates 1 anchor at each feature point, as a substitute of the
        #  grid of features.
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=1,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(_delete_=True, type='TBLRBBoxCoder', normalizer=4.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            reduction='none'),
        loss_bbox=dict(
            _delete_=True,
            type='DIoULoss',
            eps=1e-6,
            loss_weight=1.5,
            reduction='none')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            _delete_=True,
            type='CenterRegionAssigner',
            # pos_scale=0.2,
            pos_scale=0.8,
            neg_scale=0.8,
            min_pos_iof=0.01),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))