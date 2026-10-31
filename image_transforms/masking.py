"""Find objects with Grounding DINO, segment with SAM, and fill their pixels."""
from functools import lru_cache

import numpy as np
from PIL import Image

from .transforms import _prepare_image


def _word(word):
    if not isinstance(word, str) or not word.strip().strip('.'):
        raise ValueError('word must be a nonempty object name or phrase')
    return word.strip().lower().rstrip('.') + '.'


class GroundedSAM:
    """Reusable detector and segmenter; requires requirements-masking.txt.

    Defaults to CUDA when available, otherwise CPU. Models download on first
    construction. Detection thresholds control which objects get masked.
    """

    def __init__(self, device=None, box_threshold=.35, text_threshold=.25):
        for threshold in (box_threshold, text_threshold):
            if not 0 <= threshold <= 1:
                raise ValueError('thresholds must be in [0, 1]')
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        detector_id = 'IDEA-Research/grounding-dino-tiny'
        sam_id = 'facebook/sam-vit-base'
        self.detector_processor = AutoProcessor.from_pretrained(detector_id)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(self.device).eval()
        self.sam_processor = SamProcessor.from_pretrained(sam_id)
        self.sam = SamModel.from_pretrained(sam_id).to(self.device).eval()

    def segment(self, image, word):
        """Return the union of matching object masks as a boolean [H, W] array."""
        import torch

        caption = _word(word)
        image = image.convert('RGB')
        inputs = self.detector_processor(images=image, text=caption, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            outputs = self.detector(**inputs)
        detections = self.detector_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.box_threshold,
            text_threshold=self.text_threshold, target_sizes=[(image.height, image.width)],
        )[0]
        boxes = detections['boxes'].detach().cpu().numpy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image.width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image.height)
        boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
        if len(boxes) == 0:
            return np.zeros((image.height, image.width), dtype=bool)

        inputs = self.sam_processor(images=image, input_boxes=[boxes.tolist()],
                                    return_tensors='pt').to(self.device)
        with torch.inference_mode():
            outputs = self.sam(**inputs, multimask_output=False)
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs['original_sizes'].cpu(),
            inputs['reshaped_input_sizes'].cpu(),
        )[0]
        # [boxes, one mask, height, width] in original image coordinates.
        return masks[:, 0].any(dim=0).numpy().astype(bool)


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
    mask = np.asarray(segmenter.segment(source.convert('RGB'), word))
    if mask.dtype != np.bool_ or mask.shape != (source.height, source.width):
        raise ValueError('segmenter must return a boolean mask matching the image dimensions')
    pixels = np.array(source)
    pixels[mask, :3] = 255 if fill == 'white' else 0
    return Image.fromarray(pixels)
