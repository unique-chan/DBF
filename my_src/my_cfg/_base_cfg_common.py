from datetime import datetime


def get_base_ckpt_and_gpu_config(cfg, parse_args):
    cfg.data.samples_per_gpu = parse_args.SAMPLES_PER_GPU
    cfg.gpu_ids = parse_args.GPU_IDS
    cfg.device = parse_args.DEVICE
    cfg.seed = parse_args.SEED
    cfg.load_from = parse_args.LOAD_FROM
    cfg.resume_from = parse_args.RESUME_FROM
    cfg.work_dir = (f'exp-{parse_args.CONFIG_FILE.replace("/", ".").replace(".py", "")}'
                    f'-{parse_args.TAG_NAME}-SEED_{parse_args.SEED}-{datetime.now().strftime("%Y%m%d_%H%M%S")}')
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
