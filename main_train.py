# from mmdet.datasets import build_dataset -> already imported in my_src/my_trainer/train.py
# from mmdet.apis import set_random_seed   -> already imported in my_src/my_trainer/train.py
# from mmdet.apis import train_detector    -> replaced with train_detector() in my_src/my_trainer/train.py

from mmdet.models import build_detector

from my_src import *

args = Parser('train').parse_args()

set_random_seed(args.seed, deterministic=args.deterministic)
cfg = get_all_configs(args, mode='train', verbose=False)

datasets = [build_dataset(cfg.data.train)]
model = build_detector(cfg.model,
                       train_cfg=cfg.get('train_cfg'),
                       test_cfg=cfg.get('test_cfg'))
if args.init_weights:
    model.init_weights()
model.CLASSES = datasets[0].CLASSES

os.makedirs(cfg.work_dir, exist_ok=True)
cfg.dump(f'{cfg.work_dir}/cfg.py')
init_for_dynamic_backbone_freezing(args)
runner = train_detector(model, datasets, cfg, distributed=False, validate=(not args.no_validate), run_time_measure=True)
save_log_from_runner(cfg.work_dir, runner)
