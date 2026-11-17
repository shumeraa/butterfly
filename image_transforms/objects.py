"""Compose existing transform callables with explicit or model-generated masks."""
from functools import partial

import numpy as np
from PIL import Image

from . import high_level as _high
from .masking import _default_segmenter, _validate_mask, _word
from .transforms import _prepare_image, flip, rotate, scale

ANCHORS = {
    'top_left': (0, 0), 'top_center': (.5, 0), 'top_right': (1, 0),
    'center_left': (0, .5), 'center': (.5, .5), 'center_right': (1, .5),
    'bottom_left': (0, 1), 'bottom_center': (.5, 1), 'bottom_right': (1, 1),
}


def _options(transform, geometry, anchor, background, fill, allow_clipping):
    if not callable(transform):
        raise TypeError('transform must be callable')
    if not isinstance(geometry, bool) or not isinstance(allow_clipping, bool):
        raise ValueError('geometry and allow_clipping must be booleans')
    operation = transform
    while isinstance(operation, partial):
        operation = operation.func
    geometric = (flip, rotate, scale, _high.horizontal_flip, _high.vertical_flip, _high.rotate_180)
    if not geometry and any(operation is function for function in geometric):
        raise ValueError('flip/rotate/scale object edits require geometry=True')
    if not isinstance(anchor, str) or anchor not in ANCHORS:
        raise ValueError(f'anchor must be one of {tuple(ANCHORS)}')
    if background is not None and fill is not None:
        raise ValueError('supply background or fill, not both')
    if not geometry and (background is not None or fill is not None or
                         anchor != 'center' or allow_clipping):
        raise ValueError('placement/background options require geometry=True')


def _replacement(source, background, fill):
    if background is not None:
        prepared = _prepare_image(background)
        if prepared.size != source.size:
            raise ValueError('background must match the image dimensions')
        return prepared.convert(source.mode)
    if fill is not None:
        # Validate the color even if the mask later turns out to be empty.
        color = Image.new(source.mode, (1, 1), fill).getpixel((0, 0))
        return Image.new(source.mode, source.size, color)
    return None


def transform_masked(image, mask, transform, *, geometry=False, anchor='center',
                     background=None, fill=None, allow_clipping=False, **kwargs):
    """Apply an existing callable to a selected region; return a new image.

    mask is boolean [H, W]. Empty masks return an unchanged RGB/RGBA copy.
    Extra keyword arguments go directly to transform.

    Appearance mode calls transform on a copy of the full image, then copies
    only masked pixels back. The callable must preserve dimensions and alpha.
    Neighborhood operations (e.g. blur) can sample outside the selected region.

    geometry=True extracts the mask's tight bounding box as an RGBA object
    layer, calls transform once, and composites its transformed alpha at anchor.
    Use this with flip, rotate, or scale; the callable must transport alpha with
    the pixels and return RGBA. The output canvas never changes. Visible object
    clipping is rejected unless allow_clipping=True.

    Uncovered/partially covered original-object pixels require a clean full-size
    background or an explicit fill color. Only the old mask is replaced before
    compositing; other scene pixels remain untouched outside the new footprint.
    No automatic inpainting or depth/occlusion inference is performed.
    """
    _options(transform, geometry, anchor, background, fill, allow_clipping)
    source = _prepare_image(image)
    mask = _validate_mask(mask, source)
    replacement = _replacement(source, background, fill)
    if not mask.any():
        return source
    selection = Image.fromarray(mask.astype(np.uint8) * 255)

    if not geometry:
        result = transform(source.copy(), **kwargs)
        if not isinstance(result, Image.Image):
            raise TypeError('transform must return a PIL.Image.Image')
        result = _prepare_image(result)
        if result.size != source.size or result.mode != source.mode:
            raise ValueError('appearance transforms must preserve dimensions and alpha mode')
        if source.mode == 'RGBA' and result.getchannel('A').tobytes() != source.getchannel('A').tobytes():
            raise ValueError('appearance transforms must preserve alpha')
        return Image.composite(result, source, selection)

    box = selection.getbbox()
    layer = source.convert('RGBA').crop(box)
    alpha = np.array(layer.getchannel('A'))
    alpha[~mask[box[1]:box[3], box[0]:box[2]]] = 0
    layer.putalpha(Image.fromarray(alpha))
    # Remove hidden background RGB as well; Pillow's RGBA geometry resamples
    # premultiplied colors, avoiding background fringes along the object edge.
    pixels = np.array(layer)
    pixels[pixels[..., 3] == 0, :3] = 0
    layer = Image.fromarray(pixels)
    transformed = transform(layer.copy(), **kwargs)
    if not isinstance(transformed, Image.Image):
        raise TypeError('transform must return a PIL.Image.Image')
    if transformed.mode != 'RGBA' or min(transformed.size) < 1:
        raise ValueError('geometry transforms must return a nonempty RGBA object layer')
    if transformed.size == layer.size and transformed.tobytes() == layer.tobytes():
        return source

    ax, ay = ANCHORS[anchor]
    left = box[0] + round((layer.width - transformed.width) * ax)
    top = box[1] + round((layer.height - transformed.height) * ay)
    support = transformed.getchannel('A').getbbox()
    if support is not None and not allow_clipping:
        if (left + support[0] < 0 or top + support[1] < 0 or
                left + support[2] > source.width or top + support[3] > source.height):
            raise ValueError('transformed object leaves the canvas; set allow_clipping=True to clip')
    overlay = Image.new('RGBA', source.size)
    overlay.paste(transformed, (left, top))

    if replacement is None:
        original_alpha = np.asarray(source.convert('RGBA').getchannel('A'))
        uncovered = mask & (original_alpha > 0) & (np.asarray(overlay.getchannel('A')) < 255)
        if uncovered.any():
            raise ValueError('geometry exposes original-object pixels; supply a clean background or fill')
        base = source
    else:
        base = Image.composite(replacement, source, selection)
    return Image.alpha_composite(base.convert('RGBA'), overlay).convert(source.mode)


def transform_objects(image, word, transform, *, segmenter=None, instance=None,
                      geometry=False, anchor='center', background=None, fill=None,
                      allow_clipping=False, **kwargs):
    """Segment by phrase, then delegate to transform_masked without new filters.

    Appearance edits with instance=None use the union of all matches, just like
    mask_objects. Geometry defaults to the sole detected instance; multiple
    matches require an explicit nonnegative instance index (detector order).
    Explicit selection and geometry require a backend with segment_instances,
    returning boolean [N, H, W]. GroundedSAM implements this interface.

    No matches with instance=None return an unchanged copy. An explicit index
    outside the detections raises ValueError, including when no objects match.
    Pass an explicit mask to transform_masked to reuse a segmentation across
    several independently generated variants without rerunning the models.
    """
    _word(word)
    _options(transform, geometry, anchor, background, fill, allow_clipping)
    if instance is not None and (isinstance(instance, bool) or
                                 not isinstance(instance, (int, np.integer)) or instance < 0):
        raise ValueError('instance must be a nonnegative integer or None')
    source = _prepare_image(image)
    # Validate the background/fill before loading a segmentation model.
    _replacement(source, background, fill)
    backend = _default_segmenter() if segmenter is None else segmenter
    if instance is None and not geometry:
        mask = backend.segment(source.convert('RGB'), word)
    else:
        segment_instances = getattr(backend, 'segment_instances', None)
        if not callable(segment_instances):
            raise TypeError('geometry/instance selection requires segment_instances; alternatively pass a mask to transform_masked')
        masks = np.asarray(segment_instances(source.convert('RGB'), word))
        if (masks.dtype != np.bool_ or masks.ndim != 3 or
                masks.shape[1:] != (source.height, source.width)):
            raise ValueError('segment_instances must return boolean [N, H, W] masks matching the image')
        if instance is not None:
            if instance >= len(masks):
                raise ValueError('instance index is outside the detected objects')
            mask = masks[instance]
        elif len(masks) > 1:
            raise ValueError('multiple objects detected; select an instance for geometry')
        else:
            mask = masks[0] if len(masks) else np.zeros((source.height, source.width), dtype=bool)
    return transform_masked(source, mask, transform, geometry=geometry, anchor=anchor,
                            background=background, fill=fill, allow_clipping=allow_clipping, **kwargs)
