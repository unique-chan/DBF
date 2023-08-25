# Original: mmcv.runner.epoch_based_runner (usage e.g. from mmcv.runner import epoch_based_runner)
#           -> Ref.: https://mmcv.readthedocs.io/en/1.x/_modules/mmcv/runner/epoch_based_runner.html
# Modified by Yechan Kim

# import os.path as osp
# import platform
# import shutil
import time
# import warnings
from typing import Any
# from typing import Dict, List, Optional, Tuple

import torch
# from torch.utils.data import DataLoader

from mmcv.runner import BaseRunner              # Ori.: from .base_runner import BaseRunner
from mmcv.runner import RUNNERS                 # Ori.: from .builder import RUNNERS
# from mmcv.runner import save_checkpoint       # Ori.: from .checkpoint import save_checkpoint
# from mmcv.runner import get_host_info         # Ori.: from .utils import get_host_info


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

    # def dynamic_backbone_freezing(self, epoch):
    #     '''
    #     Implemented by Yechan Kim
    #     Usage:  For training with DBF, any model should own an attribute named `gate_on` as follows:
    #             Here, `gate_on` should be implemented as a boolean function with an input argument `epoch`
    #             e.g.    Class YourDetectionModel(...):
    #                         self.gate_on = lambda epoch: True if not epoch % 10 else False
    #                         ...
    #     '''
    #     assert getattr(self.model, 'gate_on'), \
    #            (f'Please modify your self.model to have an attribute named `gate_on` as a boolean function '
    #             f'with an input argument `epoch` '
    #             f'(e.g. [model] self.gate_on = lambda epoch: True if not epoch % 10 else False).')
    #     self.model.gate_on(epoch)

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        # self.dynamic_backbone_freezing(self._epoch)  # Added by Yechan Kim
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
