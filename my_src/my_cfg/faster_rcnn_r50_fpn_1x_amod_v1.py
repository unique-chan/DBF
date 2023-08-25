from mmcv import Config

from ._base_cfg_common import get_base_ckpt_and_gpu_config, get_base_log_config
from ._base_cfg_for_amod_v1 import get_base_data_config


def get_config(parse_args, verbose=True):
    PRETRAINED_MODEL_CONFIG = 'mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    cfg = Config.fromfile(PRETRAINED_MODEL_CONFIG)
    cfg = get_base_data_config(cfg, parse_args)
    cfg = get_base_ckpt_and_gpu_config(cfg, parse_args)
    cfg = get_base_log_config(cfg)
    cfg.model.roi_head.bbox_head.num_classes = 13
    # cfg.model.bbox_head.num_classes = 13
    cfg.optimizer.lr = 0.01
    cfg.runner.max_epochs = parse_args.EPOCHS
    cfg.lr_config = dict(
        policy='step',
        warmup=None,
        warmup_iters=500,
        warmup_ratio=0.001,
        step=[100]
    )
    if verbose:
        print(f'▶️ {cfg.pretty_text}')
    return cfg
