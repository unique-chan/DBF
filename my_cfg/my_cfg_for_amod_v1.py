from mmcv import Config
from mmdet.apis import set_random_seed

from my_parser import Parser

ps = Parser().parse_args()

set_random_seed(ps.seed, deterministic=False)
cfg = Config.fromfile(ps.config_file)

cfg.dataset_type = 'AMODv1'
cfg.data_root = ''
cfg.data.samples_per_gpu = ps.samples_per_gpu

cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root = f'{cfg.data_root}/train_malden'
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
cfg.data.val.data_root = f'{cfg.data_root}/test_stratis'
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
cfg.data.test.data_root = f'{cfg.data_root}/test_tanoa'
cfg.data.test.ann_file = ''
cfg.data.test.img_prefix = ''
cfg.data.test.pipeline = cfg.data.val.pipeline

cfg.model.roi_head.bbox_head.num_classes = 13
cfg.model.bbox_head.num_classes = 13
# cfg.load_from = None
# cfg.resume_from = None

cfg.work_dir = f'/exp-{ps.tag_name}-{0}'
cfg.checkpoint_config.interval = 1

cfg.optimizer.lr = 0.01
cfg.runner.max_epochs = ps.epochs
cfg.lr_config = dict(
    policy='step',
    warmup=None,
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[100]
)
cfg.log_config.interval = 1870
cfg.log_config.hooks = [dict(type='TextLoggerHook'),
                        #  dict(type='MMDetWandbHook',
                        #       init_kwargs=dict(
                        #           project=projectname,
                        #           name=tagname
                        #       ),
                        #       interval=10,
                        #       log_checkpoint=True,
                        #       log_checkpoint_metadata=True,
                        #       num_eval_images=100)
                        ]

cfg.evaluation.metric = 'mAP'  # 'bbox'
cfg.evaluation.interval = 1

cfg.gpu_ids = ps.gpu_ids
cfg.device = ps.device

cfg.seed = 0
