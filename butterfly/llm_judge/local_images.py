#!/usr/bin/env python3
"""Render deterministic MR2 edits from an image and an individual object mask."""
import argparse
import json
import math
from pathlib import Path
import random

from evaluate_rules import GLOBAL, LOCAL_TYPES, evaluate


def render_local(image, mask, transformation_type, *, answer_type, topic, objects,
                 flags, object_index=0, mr1=None, scale_x=None,
                 scale_y=None, background=None, seed=42, answer_options=()):
    """Return (PIL image, label, reason); the supplied mask must isolate one object.

    Geometry edits require a clean background plate to expose the vacated pixels.
    Semantic U edits can be rendered, but shared segmentation/target blockers cannot.
    """
    from PIL import Image, ImageChops, ImageFilter, ImageOps

    if transformation_type not in LOCAL_TYPES:
        raise ValueError('Unsupported MR2 transformation type')
    if transformation_type == 'segment_mr1' and mr1 not in GLOBAL:
        raise ValueError('segment_mr1 requires one of the six MR1 transformations')
    name = 'segment_mr1_' + mr1 if transformation_type == 'segment_mr1' else transformation_type
    _, local = evaluate(answer_type, topic, objects, flags, answer_options)
    if not isinstance(object_index, int) or not 0 <= object_index < len(local):
        raise ValueError('Invalid object_index')
    if name not in local[object_index]:
        raise ValueError('Only REFERENT objects may be edited')
    label, reason = local[object_index][name]
    if reason in {'binary_non_target', 'segment_sensitive'}:
        raise ValueError('MR2 targeting/segmentation restriction: ' + reason)
    image = image.convert('RGB')
    if mask.size != image.size:
        raise ValueError('Mask and image dimensions must match')
    # Binary membership preserves a uniform fill even with antialiased input masks.
    mask = mask.convert('L').point(lambda value: 255 if value > 127 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError('Object mask is empty')
    crop, alpha = image.crop(bbox), mask.crop(bbox)
    geometry = transformation_type == 'asymmetric_scale' or (
        transformation_type == 'segment_mr1' and mr1 in {'hflip', 'vflip', 'rot180'})
    if geometry:
        if background is None or background.size != image.size:
            raise ValueError('Geometry edits require an image-sized clean background plate')
        result = Image.composite(background.convert('RGB'), image, mask)
    else:
        result = image.copy()
    position = bbox[:2]
    if transformation_type in {'fill_red', 'fill_blue', 'occlude_bbox'}:
        color = {'fill_red': 'red', 'fill_blue': 'blue', 'occlude_bbox': 'black'}[transformation_type]
        crop = Image.new('RGB', crop.size, color)
        if transformation_type == 'occlude_bbox':
            alpha = Image.new('L', crop.size, 255)
    elif transformation_type == 'asymmetric_scale':
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
               for v in (scale_x, scale_y)) or scale_x == scale_y:
            raise ValueError('Scale factors must be finite, positive, and unequal')
        width, height = max(1, round(crop.width * scale_x)), max(1, round(crop.height * scale_y))
        if (width, height) == crop.size:
            raise ValueError('Scale factors do not change the object at this resolution')
        position = (round((bbox[0] + bbox[2] - width) / 2),
                    round((bbox[1] + bbox[3] - height) / 2))
        if position[0] < 0 or position[1] < 0 or position[0] + width > image.width or position[1] + height > image.height:
            raise ValueError('Scaled object would be clipped by image boundaries')
        # Premultiplied alpha avoids bleeding background colors into object edges.
        layer = crop.convert('RGBA')
        layer.putalpha(alpha)
        layer = layer.convert('RGBa').resize((width, height), Image.Resampling.BICUBIC).convert('RGBA')
        crop, alpha = layer.convert('RGB'), layer.getchannel('A')
    elif mr1 in {'hflip', 'vflip', 'rot180'}:
        operation = {'hflip': Image.Transpose.FLIP_LEFT_RIGHT,
                     'vflip': Image.Transpose.FLIP_TOP_BOTTOM,
                     'rot180': Image.Transpose.ROTATE_180}[mr1]
        crop, alpha = crop.transpose(operation), alpha.transpose(operation)
    elif mr1 == 'grayscale':
        crop = ImageOps.grayscale(crop).convert('RGB')
    elif mr1 == 'blur':
        # Normalize masked blur so surrounding scene pixels do not bleed in.
        weight = alpha.filter(ImageFilter.GaussianBlur(2))
        channels = []
        for channel in crop.split():
            blurred = ImageChops.multiply(channel, alpha).filter(ImageFilter.GaussianBlur(2))
            channels.append(Image.frombytes('L', crop.size, bytes(
                min(255, round(v * 255 / w)) if w else 0
                for v, w in zip(blurred.tobytes(), weight.tobytes()))))
        crop = Image.merge('RGB', channels)
    elif mr1 == 'noise':
        rng = random.Random(seed)
        crop = Image.frombytes('RGB', crop.size, bytes(
            max(0, min(255, round(v + rng.gauss(0, 15)))) for v in crop.tobytes()))
    result.paste(crop, position, alpha)
    return result, label, reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('image', 'mask', 'prediction', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--type', choices=LOCAL_TYPES, required=True)
    parser.add_argument('--object-index', type=int, default=0)
    parser.add_argument('--mr1', choices=GLOBAL)
    parser.add_argument('--scale-x', type=float)
    parser.add_argument('--scale-y', type=float)
    parser.add_argument('--background', type=Path, help='Clean background plate for geometry edits')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    from PIL import Image
    try:
        data = json.loads(args.prediction.read_text())
        with Image.open(args.image) as image, Image.open(args.mask) as mask:
            background = Image.open(args.background) if args.background else None
            try:
                result, label, reason = render_local(
                    image, mask, args.type, answer_type=data['answer_type'], topic=data['topic'],
                    objects=data['objects'], flags=data['question_flags'], object_index=args.object_index,
                    answer_options=data.get('answer_options', []),
                    mr1=args.mr1, scale_x=args.scale_x, scale_y=args.scale_y,
                    background=background, seed=args.seed)
                if args.output.resolve() in {p.resolve() for p in (args.image, args.mask, args.prediction, args.background) if p}:
                    raise ValueError('Output must differ from input paths')
                result.save(args.output)
                print(json.dumps({'transformation_type': args.type, 'mr1': args.mr1,
                                  'object_index': args.object_index, 'label': label, 'reason_code': reason,
                                  'color': {'fill_red': 'red', 'fill_blue': 'blue', 'occlude_bbox': 'black'}.get(args.type), 'scale_x': args.scale_x, 'scale_y': args.scale_y,
                                  'seed': args.seed, 'output': str(args.output)}))
            finally:
                if background is not None:
                    background.close()
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
