from datetime import datetime


def get_base_data_config(cfg, parse_args):
    cfg.dataset_type = 'AMODv1'
    cfg.data_root = parse_args.DATA_ROOT
    # train
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
    # val
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
    # test
    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = f'{cfg.data_root}/test'
    cfg.data.test.ann_file = ''
    cfg.data.test.img_prefix = ''
    cfg.data.test.pipeline = cfg.data.val.pipeline
    return cfg


def get_base_ckpt_and_gpu_config(cfg, parse_args):
    cfg.data.samples_per_gpu = parse_args.SAMPLES_PER_GPU
    cfg.gpu_ids = parse_args.GPU_IDS
    cfg.device = parse_args.DEVICE
    cfg.seed = parse_args.SEED
    cfg.load_from = parse_args.LOAD_FROM
    cfg.resume_from = parse_args.RESUME_FROM
    cfg.work_dir = (f'exp-{parse_args.CONFIG_FILE.replace("/", ".").replace(".py", "")}'
                    f'-{parse_args.TAG_NAME}-SEED{parse_args.SEED}-{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    # cfg.workflow = [('train', 1), ('val', 1)]
    cfg.checkpoint_config.interval = -1  # save only when val mAP is best
    # cfg.checkpoint_config = dict(
    #     type='BestMetricCheckpointHook',
    #     monitor_metric='mAP',
    #     mode='max',
    #     interval=1,
    #     save_optimizer=True
    # ) # -> will be removed
    return cfg


def get_base_log_config(cfg):
    cfg.log_config.interval = 100
    cfg.log_config.hooks = [dict(type='TextLoggerHook'),
                            dict(type='TensorboardLoggerHook')
                            ]
    cfg.evaluation.metric = 'mAP'  # 'bbox'
    cfg.evaluation.save_best = 'mAP'
    cfg.evaluation.interval = 1
    return cfg
