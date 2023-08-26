# from mmdet.datasets import build_dataset -> already imported in my_src/my_trainer/train.py
# from mmdet.apis import set_random_seed   -> already imported in my_src/my_trainer/train.py
# from mmdet.apis import train_detector    -> replaced with train_detector() in my_src/my_trainer/train.py

from mmdet.models import build_detector

from my_src import *

ps = Parser().parse_args()
set_random_seed(ps.SEED, deterministic=False)
cfg = get_config(ps)

datasets = [build_dataset(cfg.data.train)]
model = build_detector(cfg.model,
                       train_cfg=cfg.get('train_cfg'),
                       test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES

os.makedirs(cfg.work_dir, exist_ok=True)
init_dynamic_backbone_freezing(ps)
runner = train_detector(model, datasets, cfg, distributed=False, validate=True)
save_log_from_runner(cfg.work_dir, runner)
