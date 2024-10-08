# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmrotate.registry import HOOKS


@HOOKS.register_module()
class RFLAModeSwitchHook(Hook):
    """Switch the mode of YOLOX during training.

    This hook turns off the mosaic and mixup data augmentation and switches
    to use L1 loss in bbox_head.

    Args:
        num_last_epochs (int): The number of latter epochs in the end of the
            training to close the data augmentation and switch to L1 loss.
            Defaults to 15.
       skip_type_keys (Sequence[str], optional): Sequence of type string to be
            skip pipeline. Defaults to ('Mosaic', 'RandomAffine', 'MixUp').
    """

    def __init__(
        self,
        num_last_epochs: int = 15,
        skip_type_keys: Sequence[str] = ('Mosaic', 'RandomAffine', 'MixUp')
    ) -> None:
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = skip_type_keys
        self._restart_dataloader = False
        self._has_switched = False

    def before_train_epoch(self, runner) -> None:
        """Close mosaic and mixup augmentation and switches to use L1 loss."""
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        model = runner.model

        if is_model_wrapper(model):
            model = model.module
        epoch_to_be_switched = ((epoch + 1) > runner.max_epochs - self.num_last_epochs)
        if epoch_to_be_switched and not self._has_switched:
            runner.logger.info('Switch model.rpn_head.assigner!')
            model.rpn_head.assigner._has_switched = True
            self._has_switched = True
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True
