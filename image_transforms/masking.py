"""Find objects with Grounding DINO, segment with SAM, and fill their pixels."""
from functools import lru_cache

import numpy as np
from PIL import Image

from .transforms import _prepare_image


def _word(word):
    if not isinstance(word, str) or not word.strip().strip('.'):
        raise ValueError('word must be a nonempty object name or phrase')
    return word.strip().lower().rstrip('.') + '.'


def _validate_mask(mask, image):
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != (image.height, image.width):
        raise ValueError('segmenter must return a boolean mask matching the image dimensions')
    return mask


class GroundedSAM:
    """Reusable detector and segmenter; requires requirements-masking.txt.

    Defaults to CUDA when available, otherwise CPU. Models download on first
    construction. Detection thresholds control which objects get masked.
    """

    def __init__(self, device=None, box_threshold=.25, text_threshold=.25,
                 detector_id='IDEA-Research/grounding-dino-base'):
        for threshold in (box_threshold, text_threshold):
            if not 0 <= threshold <= 1:
                raise ValueError('thresholds must be in [0, 1]')
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        sam_id = 'facebook/sam-vit-base'
        self.detector_processor = AutoProcessor.from_pretrained(detector_id)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(self.device).eval()
        self.sam_processor = SamProcessor.from_pretrained(sam_id)
        self.sam = SamModel.from_pretrained(sam_id).to(self.device).eval()

    def segment(self, image, word):
        """Return the union of matching object masks as a boolean [H, W] array."""
        return self.segment_instances(image, word).any(axis=0)

    def detect_boxes(self, image, word, threshold=None):
        """Return matching [x0, y0, x1, y1] boxes in original-image coordinates.

        threshold overrides box_threshold for this call (used to check a result at a lower bar).
        """
        import torch

        caption = _word(word)
        bar = self.box_threshold if threshold is None else threshold
        image = image.convert('RGB')
        inputs = self.detector_processor(images=image, text=caption, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            outputs = self.detector(**inputs)
        post = self.detector_processor.post_process_grounded_object_detection
        kwargs = dict(text_threshold=self.text_threshold, target_sizes=[(image.height, image.width)])
        try:  # transformers >=4.51 names it `threshold`; older releases use `box_threshold`
            detections = post(outputs, inputs.input_ids, threshold=bar, **kwargs)[0]
        except TypeError:
            detections = post(outputs, inputs.input_ids, box_threshold=bar, **kwargs)[0]
        boxes = detections['boxes'].detach().cpu().numpy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image.width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image.height)
        boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
        return boxes

    def segment_instances(self, image, word):
        """Return boolean [N, H, W] masks in detector order, before unioning.

        Indices are not semantic identities; inspect masks to select an instance.
        No detections return an empty [0, H, W] array.
        """
        import torch

        image = image.convert('RGB')
        return self.segment_boxes(image, self.detect_boxes(image, word))

    def segment_boxes(self, image, boxes):
        """Return boolean [N, H, W] SAM masks for the given [x0, y0, x1, y1] boxes."""
        import torch

        image = image.convert('RGB')
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, image.height, image.width), dtype=bool)

        inputs = self.sam_processor(images=image, input_boxes=[boxes.tolist()],
                                    return_tensors='pt').to(self.device)
        with torch.inference_mode():
            outputs = self.sam(**inputs, multimask_output=False)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs['original_sizes'].cpu(),
            inputs['reshaped_input_sizes'].cpu(),
        )[0]
        # [boxes, one mask, height, width] in original image coordinates.
        return masks[:, 0].numpy().astype(bool)


@lru_cache(maxsize=1)
def _default_segmenter():
    return GroundedSAM()


def mask_objects(image, word, fill='white', *, segmenter=None):
    """Replace objects matching word with white (or black).

    All matches are filled. No matches return an unchanged copy. Alpha and
    unmasked pixels are preserved. Pass a GroundedSAM instance to set device
    or thresholds; otherwise the default models load once and are reused.
    """
    _word(word)
    if fill not in ('white', 'black'):
        raise ValueError("fill must be 'white' or 'black'")
    source = _prepare_image(image)
    if segmenter is None:
        segmenter = _default_segmenter()
    mask = _validate_mask(segmenter.segment(source.convert('RGB'), word), source)
    pixels = np.array(source)
    pixels[mask, :3] = 255 if fill == 'white' else 0
    return Image.fromarray(pixels)


def _tint(pixels, mask, rgb, strength):
    """Shift masked pixels to rgb's hue and toward its saturation, keeping brightness."""
    hsv = np.array(Image.fromarray(np.ascontiguousarray(pixels[:, :, :3])).convert('HSV')).astype(float)
    target = np.array(Image.new('RGB', (1, 1), rgb).convert('HSV').getpixel((0, 0)), dtype=float)
    hsv[mask, 0] = target[0]
    hsv[mask, 1] = hsv[mask, 1] * (1 - strength) + target[1] * strength
    tinted = np.array(Image.fromarray(hsv.round().astype(np.uint8), 'HSV').convert('RGB'))
    pixels[mask, :3] = tinted[mask]


def _dilate_mask(mask, fraction):
    """Grow a boolean mask by fraction of the shorter image side (covers soft edges/missed rims)."""
    from PIL import ImageFilter
    radius = int(round(fraction * min(mask.shape)))
    if radius < 1:
        return mask
    grown = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * radius + 1))
    return np.asarray(grown) > 0


DEFAULT_FILL_STRENGTH = 0.75   # gqa-simple-v5 was built with solid fill (strength=1.0); pass 1.0 to reproduce it


def fill_objects(image, word, color, *, segmenter=None, strength=DEFAULT_FILL_STRENGTH, dilate=.01):
    """Fill the matching segmentation mask with a color, preserving alpha.

    Raise if no target is found instead of recording an unchanged custom test.
    All objects matched by the supplied phrase are filled. strength=1 replaces
    every masked pixel with the color; 0<strength<1 instead tints the object
    (target hue, saturation moved toward the color's), keeping its shading.
    """
    _word(word)
    if not 0 < strength <= 1:
        raise ValueError('strength must be in (0, 1]')
    rgb = Image.new('RGB', (1, 1), color).getpixel((0, 0))
    source = _prepare_image(image)
    backend = segmenter if segmenter is not None else _default_segmenter()
    mask = _validate_mask(backend.segment(source.convert('RGB'), word), source)
    if not mask.any():
        raise ValueError(f'No object matched {word!r} for color fill')
    mask = _dilate_mask(mask, dilate)
    pixels = np.array(source)
    if strength == 1:
        pixels[mask, :3] = rgb
    else:
        _tint(pixels, mask, rgb, strength)
    return Image.fromarray(pixels)


def occlude_objects(image, word, fill='white', *, segmenter=None, dilate=.1, verify=True,
                    verify_threshold=.2, max_rounds=2, max_area=None, shape='box', fallback=None,
                    mask_dilate=.01):
    """occlude_objects_once, retried with the `fallback` shape if the first shape is rejected.

    shape/fallback are 'box' (dilated detection boxes) or 'mask' (SAM masks of the boxes, much
    tighter on thin or irregular objects such as fences and grass, so fewer tests hit max_area).
    A missing object is never retried.
    """
    options = dict(segmenter=segmenter, dilate=dilate, verify=verify, verify_threshold=verify_threshold,
                   max_rounds=max_rounds, max_area=max_area, mask_dilate=mask_dilate)
    try:
        return _occlude_once(image, word, fill, shape=shape, **options)
    except ValueError as error:
        if not fallback or fallback == shape or str(error).startswith('No object matched'):
            raise
        return _occlude_once(image, word, fill, shape=fallback, **options)


def _occlude_once(image, word, fill, *, segmenter, dilate, verify, verify_threshold, max_rounds, max_area,
                  shape, mask_dilate):
    """Fill each matching detection box (including its background) with a color.

    Custom backends without detection boxes may supply instance masks; their
    tight bounding boxes are used instead. Empty detections are an error.

    dilate grows every box by that fraction of its size (so tight boxes still cover the object's
    edges). With verify (needs a backend with detect_boxes), the edited image is searched again at
    the lower verify_threshold; detections outside the filled area are filled too, up to
    max_rounds times. If the object is still found, or the fills would cover more than max_area
    of the image (when max_area is set; off by default), ValueError is raised so no unreliable "object absent" test is recorded.
    """
    _word(word)
    rgb = Image.new('RGB', (1, 1), fill).getpixel((0, 0))
    source = _prepare_image(image)
    backend = segmenter if segmenter is not None else _default_segmenter()
    has_boxes = callable(getattr(backend, 'detect_boxes', None))
    if has_boxes:
        boxes = np.asarray(backend.detect_boxes(source.convert('RGB'), word), dtype=float)
    else:
        if callable(getattr(backend, 'segment_instances', None)):
            masks = backend.segment_instances(source.convert('RGB'), word)
        else:
            masks = [backend.segment(source.convert('RGB'), word)]
        boxes = []
        for mask in masks:
            mask = _validate_mask(mask, source)
            y, x = np.nonzero(mask)
            if len(x):
                boxes.append([x.min(), y.min(), x.max() + 1, y.max() + 1])
        boxes = np.asarray(boxes, dtype=float)
    if boxes.size == 0:
        raise ValueError(f'No object matched {word!r} for bounding-box occlusion')
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError('Detector must return finite [N, 4] bounding boxes')
    pixels = np.array(source)
    filled = np.zeros((source.height, source.width), dtype=bool)

    if shape not in ('box', 'mask'):
        raise ValueError("shape must be 'box' or 'mask'")
    if shape == 'mask' and not callable(getattr(backend, 'segment_boxes', None)):
        raise ValueError("shape='mask' needs a backend with segment_boxes")

    def paint(boxes, current):
        if shape == 'mask':
            for mask in backend.segment_boxes(current, boxes):
                region = _dilate_mask(mask, mask_dilate)
                pixels[region, :3] = rgb
                filled[region] = True
            return
        for x0, y0, x1, y1 in boxes:
            x0, y0, x1, y1 = int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))
            gx, gy = int(round(dilate * (x1 - x0) / 2)), int(round(dilate * (y1 - y0) / 2))
            x0, y0 = max(0, x0 - gx), max(0, y0 - gy)
            x1, y1 = min(source.width, x1 + gx), min(source.height, y1 + gy)
            if x1 <= x0 or y1 <= y0:
                raise ValueError('Detector returned an empty bounding box')
            pixels[y0:y1, x0:x1, :3] = rgb
            filled[y0:y1, x0:x1] = True

    paint(boxes, source.convert('RGB'))
    if verify and has_boxes:
        extra_args = {'threshold': verify_threshold} if isinstance(backend, GroundedSAM) else {}
        for round_ in range(max_rounds + 1):
            if max_area is not None and filled.mean() > max_area:
                raise ValueError(f'Occluding {word!r} would cover over {max_area:.0%} of the image')
            edited = Image.fromarray(pixels).convert('RGB')
            found = np.asarray(backend.detect_boxes(edited, word, **extra_args), dtype=float).reshape(-1, 4)
            outside = []
            for box in found:  # a box mostly on our own fill is the detector reacting to the edit
                x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
                x1, y1 = min(source.width, int(np.ceil(box[2]))), min(source.height, int(np.ceil(box[3])))
                if x1 > x0 and y1 > y0 and filled[y0:y1, x0:x1].mean() < .5:
                    outside.append(box)
            if not outside:
                break
            if round_ == max_rounds:
                raise ValueError(f'{word!r} still detected after occlusion; test is unreliable')
            paint(np.asarray(outside), edited)
    return Image.fromarray(pixels)
