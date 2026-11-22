"""Apply selected custom-test operations to the original image and save PNGs."""
import json
import os
from pathlib import Path
import tempfile
from functools import lru_cache

from PIL import Image, ImageOps

from image_transforms import apply_transform, fill_objects, occlude_objects, GroundedSAM
from unit_test_generation.metamorphic.baselines import validate_answer_rule


@lru_cache(maxsize=4)
def _configured_segmenter(device=None, box_threshold=.35, text_threshold=.25):
    return GroundedSAM(device=device, box_threshold=box_threshold, text_threshold=text_threshold)


def materialize_selected_tests(image_path, tests, output_dir, *, seed=42,
                               segmenter=None, segmenter_config=None):
    """Independently transform the source for each selected test, preserving rules."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    original = Path(image_path).resolve()
    result = []
    with Image.open(original) as source:
        image = ImageOps.exif_transpose(source).copy()
    for index, test in enumerate(tests):
        operation = test['transformation']
        parameters = dict(test.get('parameters', {}))
        validate_answer_rule(test.get('answer', ''))
        try:
            if operation in {'fill_color', 'occlude_bbox'}:
                if segmenter is None and segmenter_config:
                    segmenter = _configured_segmenter(**dict(segmenter_config))
                if operation == 'fill_color':
                    transformed = fill_objects(image, parameters['item'], parameters['color'],
                                               segmenter=segmenter,
                                               strength=parameters.get('strength', 1.0))
                else:
                    transformed = occlude_objects(image, parameters['item'],
                        fill=parameters.get('fill', 'white'), segmenter=segmenter)
            else:
                name = {'hflip': 'horizontal_flip', 'vflip': 'vertical_flip'}.get(operation, operation)
                transformed = apply_transform(image, name, seed=seed)
        except (ValueError, KeyError) as error:
            raise ValueError(f'Cannot create custom test {test["description"]!r}: {error}') from error
        # Names are based on the selected index, not untrusted object descriptions.
        path = output / f'{index:03d}.png'
        if path == original:
            raise ValueError('Transformed output would overwrite the original image')
        descriptor, temporary = tempfile.mkstemp(dir=output, prefix='.image-', suffix='.png')
        os.close(descriptor)
        try:
            transformed.save(temporary, format='PNG')
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        result.append({**test, 'image_paths': [str(path)], 'seed': seed,
                       'original_image_path': str(original)})
    # This manifest is written only after every returned image has been saved.
    (output / 'tests.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result
