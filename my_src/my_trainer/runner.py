# Original: mmcv.runner.epoch_based_runner (usage e.g. from mmcv.runner import epoch_based_runner)
#           -> Ref.: https://mmcv.readthedocs.io/en/1.x/_modules/mmcv/runner/epoch_based_runner.html
#           -> Ref.: https://biology-statistics-programming.tistory.com/142
#           -> Ref.: https://github.com/jytime/Mask_RCNN_Pytorch/issues/2
# Modified by Yechan Kim

import os.path as osp
import platform
import shutil
import warnings

import mmcv
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from mmcv.runner import save_checkpoint       # Ori.: from .checkpoint import save_checkpoint
from mmcv.runner import get_host_info         # Ori.: from .utils import get_host_info

import time

from typing import Any

import torch

from mmcv.runner import BaseRunner              # Ori.: from .base_runner import BaseRunner
from mmcv.runner import RUNNERS                 # Ori.: from .builder import RUNNERS


# @RUNNERS.register_module()
# class EpochBasedRunner(BaseRunner):
#     """Epoch-based Runner.This runner train models epoch by epoch."""
#     def run_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:
#         if self.batch_processor is not None:
#             outputs = self.batch_processor(
#                 self.model, data_batch, train_mode=train_mode, **kwargs)
#         elif train_mode:
#             outputs = self.model.train_step(data_batch, self.optimizer,
#                                             **kwargs)
#         else:
#             outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
#         if not isinstance(outputs, dict):
#             raise TypeError('"batch_processor()" or "model.train_step()"'
#                             'and "model.val_step()" must return a dict')
#         if 'log_vars' in outputs:
#             self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
#         self.outputs = outputs
#
#     def train(self, data_loader, **kwargs):
#         self.model.train()
#         self.mode = 'train'
#         self.data_loader = data_loader
#         self._max_iters = self._max_epochs * len(self.data_loader)
#         self.call_hook('before_train_epoch')
#         time.sleep(2)  # Prevent possible deadlock during epoch transition
#         for i, data_batch in enumerate(self.data_loader):
#             self.data_batch = data_batch
#             self._inner_iter = i
#             self.call_hook('before_train_iter')
#             self.run_iter(data_batch, train_mode=True, **kwargs)
#             self.call_hook('after_train_iter')
#             del self.data_batch
#             self._iter += 1
#
#         self.call_hook('after_train_epoch')
#         self._epoch += 1
#
#     @torch.no_grad()
#     def val(self, data_loader, **kwargs):
#         self.model.eval()
#         self.mode = 'val'
#         self.data_loader = data_loader
#         self.call_hook('before_val_epoch')
#         time.sleep(2)  # Prevent possible deadlock during epoch transition
#         for i, data_batch in enumerate(self.data_loader):
#             self.data_batch = data_batch
#             self._inner_iter = i
#             self.call_hook('before_val_iter')
#             self.run_iter(data_batch, train_mode=False)
#             self.call_hook('after_val_iter')
#             del self.data_batch
#         self.call_hook('after_val_epoch')


@RUNNERS.register_module(force=True)
class EpochBasedRunnerForDBF(BaseRunner):
    fn_for_dynamic_backbone_freezing = None                   # Added by Yechan Kim
    param_for_dynamic_backbone_freezing = None                # Added by Yechan Kim

    @classmethod
    def set_fn_for_dynamic_backbone_freezing(cls, fn):        # Added by Yechan Kim
        cls.fn_for_dynamic_backbone_freezing = fn

    @classmethod
    def set_param_for_dynamic_backbone_freezing(cls, fn):     # Added by Yechan Kim
        cls.fn_for_dynamic_backbone_freezing = fn

    def run_iter(self, data_batch: Any, train_mode: bool, **kwargs) -> None:
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        if self.fn_for_dynamic_backbone_freezing is not None:  # Added by Yechan Kim
            # print(self.model.module.bool_freeze_backbone)
            EpochBasedRunnerForDBF.fn_for_dynamic_backbone_freezing(runner=self)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')
            del self.data_batch
        self.call_hook('after_val_epoch')

    def save_checkpoint(self,
                        out_dir: str,
                        filename_tmpl: str = 'epoch_{}.pth',
                        save_optimizer: bool = True,
                        meta: Optional[Dict] = None,
                        create_symlink: bool = True) -> None:
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

    def run(self, data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_epochs: Optional[int] = None,
            **kwargs) -> None:
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
