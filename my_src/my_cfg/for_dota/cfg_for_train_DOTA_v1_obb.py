def get_config(cfg, args):
    cfg.samples_per_gpu = 2
    cfg.optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
    cfg.optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    cfg.lr_config = dict(
        policy='step',
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=1.0 / 3,
        step=[8, 11])
    # cfg.workflow = [('train', 1), ('val', 1)]
    cfg.checkpoint_config.interval = -1  # save only when val mAP is best
    cfg.log_config.interval = 100
    cfg.log_config.hooks = [dict(type='TextLoggerHook'),
                            dict(type='TensorboardLoggerHook')]
    cfg.evaluation.metric = 'mAP'
    cfg.evaluation.save_best = 'mAP'
    cfg.evaluation.interval = 1
    return cfg
