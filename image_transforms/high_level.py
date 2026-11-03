"""Fixed-parameter operations used by the GQA reliability prompt.

The six names and blur/noise strengths match PROMPT and transform_images in
origin/llm_reliability_experiment:butterfly/gqa_transform_pilot/gqa_transform_pilot.py.
Each operation returns a new image using the low-level RGB/RGBA conventions.
These are independent variants of the source, not a sequence of edits.
"""
from . import transforms as _low

BLUR_RADIUS = 2
NOISE_AMOUNT = .05  # Gaussian standard deviation = .05 * 255 = 12.75.


def horizontal_flip(image):
    """Mirror left-to-right."""
    return _low.flip(image, 'left')


def vertical_flip(image):
    """Mirror top-to-bottom."""
    return _low.flip(image, 'up')


def rotate_180(image):
    """Rotate 180 degrees, preserving the canvas dimensions."""
    return _low.rotate(image, 180)


def grayscale(image):
    """Remove color using luminance grayscale, not HSV desaturation."""
    return _low.color_filter(image, 'grayscale')


def blur(image):
    """Apply Gaussian blur with a fixed radius of 2 pixels."""
    return _low.blur(image, BLUR_RADIUS)


def noise(image, *, seed=None):
    """Add Gaussian RGB noise with a fixed standard deviation of 12.75.

    Pass a nonnegative integer seed for reproducible experiment images.
    """
    return _low.noise(image, NOISE_AMOUNT, seed=seed)


_OPERATIONS = {
    'horizontal_flip': horizontal_flip,
    'vertical_flip': vertical_flip,
    'rotate_180': rotate_180,
    'grayscale': grayscale,
    'blur': blur,
    'noise': noise,
}
TRANSFORMS = tuple(_OPERATIONS)


def apply_transform(image, name, *, seed=None):
    """Apply one exact prompt key; unknown names raise ValueError.

    seed is used only by noise. Strengths are intentionally fixed to the prompt;
    use image_transforms.transforms for arbitrary angles/radii/HSV deltas, etc.
    """
    if not isinstance(name, str) or name not in _OPERATIONS:
        raise ValueError(f'name must be one of {TRANSFORMS}')
    if name == 'noise':
        return noise(image, seed=seed)
    return _OPERATIONS[name](image)


def transform_images(image, seed=None):
    """Return {prompt_key: new_image} for all six independent variants.

    Output order matches the prompt. Supply the pilot's per-image noise_seed for
    reproducibility. Alpha is preserved, unlike the pilot's RGB-only export;
    pass image.convert('RGB') when that export behavior is required.
    """
    return {name: apply_transform(image, name, seed=seed) for name in TRANSFORMS}
