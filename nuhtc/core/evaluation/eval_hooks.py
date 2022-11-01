from mmcv.runner import EvalHook as BaseEvalHook
from mmcv.engine import single_gpu_test as mmcv_single_gpu_test

class EvalHook(BaseEvalHook):

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return
        if self.test_fn is mmcv_single_gpu_test:
            from mmdet.apis import single_gpu_test
            results = single_gpu_test(runner.model, self.dataloader, show=False)
        else:
            results = self.test_fn(runner.model, self.dataloader, show=False)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)