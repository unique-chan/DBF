from mmcv import Config
from mmdet.apis import set_random_seed


def get_config_amod_v1(parser, verbose=True):
    set_random_seed(parser.SEED, deterministic=False)
    cfg = Config.fromfile(parser.CONFIG_FILE)

    cfg.dataset_type = 'AMODv1'
    cfg.data_root = ''
    cfg.data.samples_per_gpu = parser.SAMPLES_PER_GPU

    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = f'{cfg.data_root}/train'
    cfg.data.train.ann_file = ''
    cfg.data.train.img_prefix = ''
    cfg.data.train.pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', mean=[100, 100, 100], std=[50, 50, 50], to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = f'{cfg.data_root}/val'
    cfg.data.val.ann_file = ''
    cfg.data.val.img_prefix = ''
    cfg.data.val.pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(640, 480),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', mean=[100, 100, 100], std=[50, 50, 50], to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ]

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = f'{cfg.data_root}/test'
    cfg.data.test.ann_file = ''
    cfg.data.test.img_prefix = ''
    cfg.data.test.pipeline = cfg.data.val.pipeline

    cfg.model.roi_head.bbox_head.num_classes = 13
    # cfg.model.bbox_head.num_classes = 13
    cfg.load_from = parser.LOAD_FROM
    cfg.resume_from = parser.RESUME_FROM

    cfg.work_dir = f'/exp-{parser.TAG_NAME}-{0}'
    cfg.checkpoint_config = dict(
        type='BestMetricCheckpointHook',
        monitor_metric='mAP',
        mode='max',
        interval=1,
        save_optimizer=True
    )
    cfg.optimizer.lr = 0.01
    cfg.runner.max_epochs = parser.EPOCHS
    cfg.lr_config = dict(
        policy='step',
        warmup=None,
        warmup_iters=500,
        warmup_ratio=0.001,
        step=[100]
    )
    cfg.log_config.interval = 100
    cfg.log_config.hooks = [dict(type='TextLoggerHook'),
                            dict(type='TensorboardLoggerHook')
    ]

    cfg.evaluation.metric = 'mAP'  # 'bbox'
    cfg.evaluation.save_best = 'mAP'
    cfg.evaluation.interval = 1

    cfg.gpu_ids = parser.GPU_IDS
    cfg.device = parser.DEVICE

    cfg.seed = 0

    if verbose:
        print(f'▶️ {cfg.pretty_text}')

    return cfg
