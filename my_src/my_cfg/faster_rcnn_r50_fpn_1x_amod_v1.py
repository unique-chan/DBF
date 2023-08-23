from mmcv import Config

from .common_cfg_for_amod_v1 import get_data_config


def get_config(parse_args, verbose=True):
    PRETRAINED_MODEL_CONFIG = 'mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    cfg = Config.fromfile(PRETRAINED_MODEL_CONFIG)
    cfg = get_data_config(cfg, parse_args.DATA_ROOT)
    cfg.data.samples_per_gpu = parse_args.SAMPLES_PER_GPU
    cfg.model.roi_head.bbox_head.num_classes = 13
    # cfg.model.bbox_head.num_classes = 13
    cfg.load_from = parse_args.LOAD_FROM
    cfg.resume_from = parse_args.RESUME_FROM
    cfg.work_dir = f'exp-{parse_args.CONFIG_FILE.replace("/", ".").replace(".py", "")}-{parse_args.TAG_NAME}-{0}'
    cfg.checkpoint_config = dict(
        type='BestMetricCheckpointHook',
        monitor_metric='mAP',
        mode='max',
        interval=1,
        save_optimizer=True
    )
    cfg.optimizer.lr = 0.01
    cfg.runner.max_epochs = parse_args.EPOCHS
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
    cfg.gpu_ids = parse_args.GPU_IDS
    cfg.device = parse_args.DEVICE
    cfg.seed = parse_args.SEED
    # cfg.workflow = [('train', 1), ('val', 1)]
    if verbose:
        print(f'▶️ {cfg.pretty_text}')
    return cfg
