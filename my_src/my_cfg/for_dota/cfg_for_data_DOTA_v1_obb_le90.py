def get_config(cfg, args):
    num_classes = len(('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                       'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                       'basketball-court', 'storage-tank', 'soccer-ball-field',
                       'roundabout', 'harbor', 'swimming-pool', 'helicopter'))
    angle_version = 'le90'
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

    cfg.dataset_type = 'DOTAv1OBB'
    cfg.data_root = args.data_root
    # train
    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = f'{cfg.data_root}/train'
    cfg.data.train.ann_file = 'annfiles'
    cfg.data.train.img_prefix = 'images'
    cfg.data.train.pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RResize', img_scale=(1024, 1024)),
        dict(
            type='RRandomFlip',
            flip_ratio=[0.25, 0.25, 0.25],
            direction=['horizontal', 'vertical', 'diagonal'],
            version=angle_version),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]
    # val
    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = f'{cfg.data_root}/val'
    cfg.data.val.ann_file = 'annfiles'
    cfg.data.val.img_prefix = 'images'
    # cfg.data.val.pipeline = [
    #     dict(type='LoadImageFromFile'),
    #     dict(
    #         type='MultiScaleFlipAug',
    #         img_scale=(640, 480),
    #         flip=False,
    #         transforms=[
    #             dict(type='Resize', keep_ratio=True),
    #             dict(type='RandomFlip'),
    #             dict(type='Normalize', mean=[100, 100, 100], std=[50, 50, 50], to_rgb=True),
    #             dict(type='Pad', size_divisor=32),
    #             dict(type='ImageToTensor', keys=['img']),
    #             dict(type='Collect', keys=['img'])
    #         ])
    # ]
    # test
    # cfg.data.test.type = cfg.dataset_type
    # cfg.data.test.data_root = f'{cfg.data_root}/test'
    # cfg.data.test.ann_file = ''
    # cfg.data.test.img_prefix = ''
    # cfg.data.test.pipeline = cfg.data.val.pipeline
    # set number of classes for head
    try:
        cfg.model.bbox_head.num_classes = num_classes
    except:
        pass
    try:
        cfg.model.roi_head.bbox_head.num_classes = num_classes
    except:
        pass
    try:
        for _ in cfg.model.roi_head.bbox_head:
            _['num_classes'] = num_classes
    except:
        pass
    return cfg
