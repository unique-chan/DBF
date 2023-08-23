from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner import save_checkpoint


# Ref1: https://mmdetection.readthedocs.io/en/v2.28.2/tutorials/customize_runtime.html#customize-self-implemented-hooks
# Ref2: https://mmcv.readthedocs.io/en/master/_modules/mmcv/runner/hooks/checkpoint.html
@HOOKS.register_module(force=True)
class BestMetricCheckpointHook(Hook):
    def __init__(self, interval=-1, save_optimizer=True, out_dir=None, monitor_metric='mAP', mode='max', **kwargs):
        assert mode in ['min', 'max'], ValueError(f'Unsupported mode: "{mode}". '
                                                  f'Use `min` or `max` instead, please.')
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.args = kwargs
        self.best_metric = float('-inf') if self.mode == 'max' else float('inf')

    def after_train_epoch(self, runner):
        if self.every_n_epochs(runner, self.interval):
            if not self.out_dir:
                self.out_dir = runner.work_dir
            current_metric = runner.log_buffer.output[self.monitor_metric]
            improved = (self.mode == 'max' and current_metric > self.best_metric) or \
                       (self.mode == 'min' and current_metric < self.best_metric)
            if improved:
                self.best_metric = current_metric
                filename = f'best_{self.monitor_metric}.pth'
                filepath = f'{self.out_dir}/{filename}'
                optimizer = runner.optimizer if self.save_optimizer else None
                # runner.save_checkpoint(self.out_dir, save_optimizer=self.save_optimizer, **self.args)
                save_checkpoint(runner.model, filepath, optimizer=optimizer)
