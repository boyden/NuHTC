from mmcv.parallel import is_module_wrapper
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.hooks.logger import WandbLoggerHook

@HOOKS.register_module()
class WandbLoggerHook_Cus(WandbLoggerHook):

    @master_only
    def before_run(self, runner):
        super(WandbLoggerHook, self).before_run(runner)
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()
        # self.wandb.watch(runner.model.module, log='all', log_freq=10)
