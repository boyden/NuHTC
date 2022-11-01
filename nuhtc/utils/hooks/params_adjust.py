from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class FineTune(Hook):
    def __init__(self, iter=2000,):
        self.iter = iter

    def before_train_iter(self, runner):
        curr_step = runner.iter
        if curr_step == self.iter:
            for param in runner.model.module.backbone.parameters():
                param.requires_grad = True
