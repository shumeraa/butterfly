"""Image transformations using Pillow and NumPy.

Transforms return new RGB/RGBA images. Color filters leave alpha unchanged;
flips, rotations, and scaling transform it with the image.
"""
import math
from numbers import Real

import numpy as np
from PIL import Image, ImageFilter, ImageOps

COLOR_PRESETS = ('grayscale', 'sepia', 'invert', 'warm', 'cool')


def _prepare_image(image):
    if not isinstance(image, Image.Image):
        raise TypeError('image must be a PIL.Image.Image')
    has_alpha = 'A' in image.getbands() or 'transparency' in image.info
    return image.convert('RGBA' if has_alpha else 'RGB')


def _validate_number(name, value, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if low is not None and value < low:
        raise ValueError(f'{name} must be >= {low}')
    if high is not None and value > high:
        raise ValueError(f'{name} must be <= {high}')
    return float(value)


def _from_rgb(source, rgb):
    result = Image.fromarray(np.clip(np.rint(rgb), 0, 255).astype(np.uint8))
    if source.mode == 'RGBA':
        result.putalpha(source.getchannel('A'))
    return result


def flip(image, direction):
    """Reflect vertically ('up') or horizontally ('left')."""
    if direction not in ('up', 'left'):
        raise ValueError("direction must be 'up' or 'left'")
    method = (Image.Transpose.FLIP_TOP_BOTTOM if direction == 'up'
              else Image.Transpose.FLIP_LEFT_RIGHT)
    return _prepare_image(image).transpose(method)


def rotate(image, angle, *, expand=True):
    """Rotate by signed degrees (positive = counterclockwise).

    Expand the canvas unless expand=False. Fill is black for RGB, clear for RGBA.
    """
    angle = _validate_number('angle', angle)
    if not isinstance(expand, bool):
        raise ValueError('expand must be a boolean')
    return _prepare_image(image).rotate(
        angle % 360, resample=Image.Resampling.BICUBIC, expand=expand
    )


def scale(image, scale_x=1, scale_y=None):
    """Resize with positive X/Y factors, using bicubic resampling.

    Omit scale_y for uniform scaling. Dimensions round to the nearest integer
    (at least one pixel). RGBA pixels and alpha resize together. This is a shared
    primitive, deliberately not a member of the six-filter MR1 registry.
    """
    sx = _validate_number('scale_x', scale_x)
    sy = sx if scale_y is None else _validate_number('scale_y', scale_y)
    if sx <= 0 or sy <= 0:
        raise ValueError('scale factors must be > 0')
    source = _prepare_image(image)
    dimensions = (source.width * sx, source.height * sy)
    if any(not math.isfinite(value) or value > 2**31 - 1 for value in dimensions):
        raise ValueError('scaled dimensions are too large')
    size = tuple(max(1, round(value)) for value in dimensions)
    return source.resize(size, resample=Image.Resampling.BICUBIC)


def color_filter(image, which):
    """Apply a preset from COLOR_PRESETS.

    Warm adds (20, 0, -20) to RGB; cool adds (-20, 0, 20), with clipping.
    """
    if which not in COLOR_PRESETS:
        raise ValueError(f'which must be one of {COLOR_PRESETS}')
    source = _prepare_image(image)
    rgb = source.convert('RGB')
    if which == 'grayscale':
        values = np.asarray(ImageOps.grayscale(rgb).convert('RGB'))
    elif which == 'invert':
        values = 255 - np.asarray(rgb)
    else:
        values = np.asarray(rgb, dtype=np.float64)
        if which == 'sepia':
            values = values @ np.array([[.393, .769, .189],
                                       [.349, .686, .168],
                                       [.272, .534, .131]]).T
        else:
            values += (20, 0, -20) if which == 'warm' else (-20, 0, 20)
    return _from_rgb(source, values)


def blur(image, radius):
    """Gaussian blur with a nonnegative radius in pixels; alpha is unchanged."""
    radius = _validate_number('radius', radius, 0)
    source = _prepare_image(image)
    rgb = source.convert('RGB').filter(ImageFilter.GaussianBlur(radius))
    if source.mode == 'RGBA':
        rgb.putalpha(source.getchannel('A'))
    return rgb


def noise(image, amount, *, seed=None):
    """Add Gaussian RGB noise with standard deviation amount * 255.

    amount is in [0, 1]. Set seed for repeatable results.
    """
    amount = _validate_number('amount', amount, 0, 1)
    if seed is not None and (isinstance(seed, bool) or
                            not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError('seed must be a nonnegative integer or None')
    source = _prepare_image(image)
    rgb = np.asarray(source.convert('RGB'), dtype=np.float64)
    rng = np.random.default_rng(seed)
    return _from_rgb(source, rgb + rng.normal(0, amount * 255, rgb.shape))


def hsv(image, delta_h=0, delta_s=0, delta_v=0):
    """Add HSV deltas: hue in degrees; saturation/value in [-1,1].

    Hue wraps, S/V clamp to [0,1], and output RGB rounds to nearest uint8.
    """
    dh = _validate_number('delta_h', delta_h) / 360
    ds = _validate_number('delta_s', delta_s, -1, 1)
    dv = _validate_number('delta_v', delta_v, -1, 1)
    source = _prepare_image(image)
    if dh % 1 == 0 and ds == 0 and dv == 0:
        return source
    rgb = np.asarray(source.convert('RGB'), dtype=np.float64) / 255
    maximum, minimum = rgb.max(axis=-1), rgb.min(axis=-1)
    span = maximum - minimum
    saturation = np.divide(span, maximum, out=np.zeros_like(span), where=maximum != 0)
    offsets = np.divide(maximum[..., None] - rgb, span[..., None],
                        out=np.zeros_like(rgb), where=span[..., None] != 0)
    rc, gc, bc = np.moveaxis(offsets, -1, 0)
    hue = np.select([rgb[..., 0] == maximum, rgb[..., 1] == maximum],
                    [bc - gc, 2 + rc - bc], default=4 + gc - rc) / 6
    hue = np.where(span == 0, 0, hue)
    hue = (hue + dh) % 1
    saturation = np.clip(saturation + ds, 0, 1)
    value = np.clip(maximum + dv, 0, 1)
    # HSV to RGB via the six triangular channel waveforms.
    k = (hue[..., None] * 6 + np.array([5, 3, 1])) % 6
    channels = np.clip(np.minimum(k, 4 - k), 0, 1)
    result = value[..., None] * (1 - saturation[..., None] * channels)
    return _from_rgb(source, result * 255)
