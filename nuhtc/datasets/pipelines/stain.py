"""Stain-space transforms: colour deconvolution and RandStainNA augmentation.

Both work in the Ruifrok-Johnston haematoxylin / eosin / DAB space reached by
``skimage.color.rgb2hed``, and both have to undo the pipeline's channel order
first: ``LoadImageFromFile`` goes through ``mmcv.imfrombytes``, which returns
BGR, and ``to_rgb`` only happens later in ``Normalize``. Feeding BGR to
``rgb2hed`` silently applies the stain matrix to the wrong channels.
"""

import cv2
import numpy as np
from mmdet.datasets import PIPELINES
from skimage.color import hed2rgb, rgb2hed

COLOR_SPACES = ('HED', 'LAB', 'HSV')
DISTRIBUTIONS = ('normal', 'laplace', 'uniform')


def as_rgb_unit(img, bgr):
    """Float32 RGB scaled to [0, 1], the transmittance rgb2hed expects."""
    # cvtColor only accepts 8U, 16U and 32F, and the pipeline hands over either
    # uint8 or float32, so cast first and the conversion is always defined
    arr = np.asarray(img, dtype=np.float32)
    if bgr:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr / 255.0


def as_pipeline_img(rgb, dtype, bgr):
    """Inverse of :func:`as_rgb_unit`.

    Clipping is load-bearing rather than defensive: a boosted density can leave
    hed2rgb above 1.0, and casting that to uint8 wraps bright pixels to black.
    """
    out = np.clip(rgb, 0.0, 1.0).astype(np.float32) * 255.0
    if bgr:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out.astype(dtype, copy=False)


@PIPELINES.register_module()
class StainDeconv:
    """Rebuild the image from a subset of its Ruifrok-Johnston stain densities.

    Motivation is the IHC side: on a strong DAB membrane stain the nuclei are
    still there but buried under brown, and dropping the DAB density recovers
    them. Before relying on that at inference it is worth knowing what the same
    operation costs on the H&E data the model was trained on, which is what this
    transform makes measurable -- put it in the test pipeline and the evaluation
    runs on H-only (or any other combination) input.

    Note that on H&E the DAB channel is not empty: the fixed stain matrix
    projects part of the true eosin and haemoglobin density onto it, so
    ``keep_d=0`` removes real signal and shifts the overall hue.

    Args:
        keep_h (float): multiplier on the haematoxylin density.
        keep_e (float): multiplier on the eosin density. 0 discards eosin, which
            on H&E removes all the cytoplasmic context.
        keep_d (float): multiplier on the DAB density.
        prob (float): probability of applying the transform. Leave at 1.0 to
            measure the deconvolved input; lower it to train on a mix of
            deconvolved and untouched patches.
        bgr (bool): whether the incoming image is BGR, as it is everywhere in an
            mmdet pipeline before ``Normalize``.
    """

    def __init__(self, keep_h=1.0, keep_e=0.0, keep_d=0.0, prob=1.0, bgr=True):
        self.keep_h = float(keep_h)
        self.keep_e = float(keep_e)
        self.keep_d = float(keep_d)
        self.prob = float(prob)
        self.bgr = bgr

    def __call__(self, results):
        if np.random.random() >= self.prob:
            return results
        for key in results.get('img_fields', ['img']):
            img = results[key]
            hed = rgb2hed(as_rgb_unit(img, self.bgr))
            hed = np.stack([
                hed[..., 0] * self.keep_h,
                hed[..., 1] * self.keep_e,
                hed[..., 2] * self.keep_d,
            ], axis=-1)
            results[key] = as_pipeline_img(hed2rgb(hed), img.dtype, self.bgr)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(keep_h={self.keep_h}, '
                f'keep_e={self.keep_e}, keep_d={self.keep_d}, '
                f'prob={self.prob}, bgr={self.bgr})')


@PIPELINES.register_module()
class RandStainNA:
    """Reinhard-style stain augmentation with a randomly drawn target template.

    Port of https://github.com/yiqings/RandStainNA (MICCAI 2022) as a pipeline
    transform. Each call samples a per-channel target mean and standard
    deviation from distributions fitted on the training set, then linearly
    remaps the image's own channel statistics onto them. All three channels are
    kept, so unlike :class:`StainDeconv` this models "the same tissue stained
    darker or bluer", not "this stain is absent".

    The two are complementary: this one makes the model tolerant of the hue
    shift that dominates StainDeconv's cost on H&E, while StainDeconv is what
    actually strips a DAB layer at inference.

    ``stats`` maps each channel to ``avg`` and ``std`` entries, themselves
    ``mean``/``std`` pairs describing how that statistic varies across the
    training set, i.e. the layout of the upstream yaml::

        stats=dict(
            color_space='HED',
            channels=(dict(avg=dict(mean=..., std=...),
                           std=dict(mean=..., std=...)), ...))

    Args:
        stats (dict): the template statistics, laid out as above.
        yaml_file (str, optional): read ``stats`` from an upstream yaml instead.
        std_hyper (float): widens (>0) or narrows (<0) the sampling spread.
        distribution (str): 'normal', 'laplace' or 'uniform'.
        prob (float): probability of applying the transform at all.
        hed_rescale (bool): the reference implementation min-max stretches the
            reconstructed HED image per patch. That is adaptive to each image's
            own extremes, which fights a fixed ``img_norm_cfg``, so it is
            available but off by default.
        bgr (bool): whether the incoming image is BGR.
    """

    def __init__(self,
                 stats=None,
                 yaml_file=None,
                 std_hyper=0.0,
                 distribution='normal',
                 prob=1.0,
                 hed_rescale=False,
                 bgr=True):
        assert distribution in DISTRIBUTIONS, \
            f'unsupported distribution {distribution}'
        assert (stats is None) != (yaml_file is None), \
            'give exactly one of stats or yaml_file'
        if yaml_file is not None:
            stats = self.load_yaml(yaml_file)
        self.color_space = stats['color_space'].upper()
        assert self.color_space in COLOR_SPACES, \
            f'unsupported color space {self.color_space}'
        channels = stats['channels']
        assert len(channels) == 3, 'need statistics for three channels'
        self.avg_mean = np.array([c['avg']['mean'] for c in channels])
        self.avg_std = np.array([c['avg']['std'] for c in channels])
        self.std_mean = np.array([c['std']['mean'] for c in channels])
        self.std_std = np.array([c['std']['std'] for c in channels])
        self.std_hyper = float(std_hyper)
        self.distribution = distribution
        self.prob = float(prob)
        self.hed_rescale = hed_rescale
        self.bgr = bgr

    @staticmethod
    def load_yaml(path):
        """Read the upstream yaml, whose channels are keyed by their letters."""
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        space = cfg['color_space']
        return dict(color_space=space,
                    channels=[cfg[letter] for letter in space])

    def sample_template(self):
        """Draw a target mean and standard deviation for each channel."""
        if self.distribution == 'uniform':
            # the three-sigma rule, as in the reference implementation
            avg = np.random.uniform(self.avg_mean - 3 * self.avg_std,
                                    self.avg_mean + 3 * self.avg_std)
            std = np.random.uniform(self.std_mean - 3 * self.std_std,
                                    self.std_mean + 3 * self.std_std)
            return avg, std
        draw = (np.random.normal
                if self.distribution == 'normal' else np.random.laplace)
        spread = 1.0 + self.std_hyper
        avg = draw(self.avg_mean, self.avg_std * spread)
        std = draw(self.std_mean, self.std_std * spread)
        return avg, std

    def to_space(self, img):
        """Pipeline image to the working colour space."""
        if self.color_space == 'HED':
            return rgb2hed(as_rgb_unit(img, self.bgr))
        # LAB and HSV are defined on 8-bit input in OpenCV's conventions
        rgb = as_pipeline_img(
            as_rgb_unit(img, self.bgr), np.uint8, bgr=False)
        code = cv2.COLOR_RGB2LAB if self.color_space == 'LAB' \
            else cv2.COLOR_RGB2HSV
        return cv2.cvtColor(rgb, code).astype(np.float32)

    def from_space(self, arr, dtype):
        """Inverse of :meth:`to_space`, back to the pipeline's channel order."""
        if self.color_space == 'HED':
            rgb = hed2rgb(arr)
            if self.hed_rescale:
                lo, hi = rgb.min(), rgb.max()
                rgb = (rgb - lo) / max(hi - lo, 1e-8)
            return as_pipeline_img(rgb, dtype, self.bgr)
        code = cv2.COLOR_LAB2RGB if self.color_space == 'LAB' \
            else cv2.COLOR_HSV2RGB
        rgb = cv2.cvtColor(
            np.clip(arr, 0, 255).astype(np.uint8), code)
        return as_pipeline_img(rgb / 255.0, dtype, self.bgr)

    def __call__(self, results):
        if np.random.random() >= self.prob:
            return results
        tar_avg, tar_std = self.sample_template()
        for key in results.get('img_fields', ['img']):
            img = results[key]
            arr = self.to_space(img)
            avg = arr.mean(axis=(0, 1))
            # a flat channel would blow up the ratio; the reference clips it
            std = np.clip(arr.std(axis=(0, 1)), 1e-4, None)
            arr = (arr - avg) * (tar_std / std) + tar_avg
            results[key] = self.from_space(arr, img.dtype)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(color_space={self.color_space}, '
                f'distribution={self.distribution}, '
                f'std_hyper={self.std_hyper}, prob={self.prob}, '
                f'hed_rescale={self.hed_rescale}, bgr={self.bgr})')
