"""Materialize LLM-selected transformations and append custom test records."""

import hashlib
import json
import sys
import random
from pathlib import Path
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageOps
from image_transforms import HIGH_LEVEL_TRANSFORMS, apply_transform
from unit_test_generation.metamorphic.baselines import validate_answer_rule


def materialize_tests(image_path, transformations, output_dir, *, num_tests=2, seed=0):
    """Create independent, seeded image variants with blank expected answers."""
    names = list(dict.fromkeys(transformations))
    if not names or any(name not in HIGH_LEVEL_TRANSFORMS for name in names):
        raise ValueError(f"Transformations must be a nonempty selection of {HIGH_LEVEL_TRANSFORMS}")
    if not isinstance(num_tests, int) or isinstance(num_tests, bool) or num_tests <= 0:
        raise ValueError("num_tests must be a positive integer")
    selected = random.Random(seed).sample(names, min(num_tests, len(names)))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tests = []
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        for name in selected:
            result = apply_transform(image, name, seed=seed)
            path = output / f"{Path(image_path).stem}_{name}.png"
            result.save(path)
            tests.append(dict(description=name, image_paths=[str(path)], answer=""))
    return tests


def eligible_transformations(parsed):
    """Use the merged prediction format, excluding unsafe decisions."""
    return [name for name, decision in parsed.items()
            if decision["relation"] != "unsafe"]


def append_custom_unit_tests(dataset_data: list, tests_by_question_id: dict) -> int:
    """Append custom tests in place after ViUniT selects tests and gets images.

    Requires unit_test_generation.use_program=False. Mapping keys identify
    questionId (or sample_id if questionId is absent), never imageId. Values:
    [{"description": "Flipped car", "answer": "red",
      "image_paths": ["tests/car_flipped.jpg"]}]

    Empty answers preserve the baseline; dictionaries map the baseline answer. Every supplied case is selected. Exact description/answer/path duplicates
    are skipped so this can be called in each feedback iteration. Paths must
    be accessible on the machine running the pipeline. Returns cases added.
    """
    entries = {}
    for entry in dataset_data:
        question_id = entry.get("questionId", entry.get("sample_id"))
        if question_id is not None:
            key = str(question_id)
            if key in entries:
                raise ValueError(f"Duplicate question ID: {key}")
            entries[key] = entry

    # Validate all additions before modifying the dataset.
    pending = []
    for question_id, custom_tests in tests_by_question_id.items():
        key = str(question_id)
        if key not in entries:
            raise ValueError(f"Unknown question ID: {key}")
        entry = entries[key]
        selected = entry.get("selected_unit_tests")
        images = entry.get("unit_test_images")
        if not isinstance(selected, list) or not isinstance(images, list):
            raise ValueError(
                "Obtain selected tests/images first and use use_program=False"
            )
        if len(selected) != len(images):
            raise ValueError(f"Tests and image groups do not align for {key}")
        for test in custom_tests:
            description, answer = test["description"], test["answer"]
            paths = test["image_paths"]
            if not isinstance(description, str) or not description.strip():
                raise ValueError("Each custom test needs a nonempty description")
            validate_answer_rule(answer)
            if not isinstance(paths, list) or not paths:
                raise ValueError("image_paths must be a nonempty list")
            if any(not isinstance(p, (str, Path)) or not str(p) for p in paths):
                raise ValueError("Image paths must be nonempty strings or Paths")
            pending.append((entry, (description, answer), [str(Path(p).resolve()) for p in paths]))

    added = 0
    for entry, test, paths in pending:
        already_present = any(
            tuple(existing_test) == test and
            [str(Path(path).resolve()) for path in existing_images] == paths
            for existing_test, existing_images in zip(
                entry["selected_unit_tests"], entry["unit_test_images"]
            )
        )
        if not already_present:
            metadata = entry.setdefault('unit_test_metadata', [
                {'source': 'viunit'} for _ in entry['selected_unit_tests']
            ])
            entry["selected_unit_tests"].append(test)
            entry["unit_test_images"].append(paths)
            metadata.append({'source': 'metamorphic'})
            added += 1
    return added

# The new branch format exports test manifests. Its image operations are still
# placeholders upstream; this helper does not materialize those image paths.
def create_test_manifest(json_string: str, img: str, out: str, out_json: str) -> dict:
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    try:
        obj = json.loads(json_string) # obj is a list
        root_out_dir = Path(out)
        img_Path = Path(img)
        res = dict() # questionID: list of tests
        for q in obj:
            # q is a list
            question_id = q[0]
            llm_output = q[1]

            with Image.open(img) as source:
                image = ImageOps.exif_transpose(source)
                for transformation_name, effect in llm_output.items():
                    # if the transformation is unsafe, skip this transformation
                    if effect == 'unsafe' or not effect: # not effect checks for empty mr2 dict
                        continue
                    # mr2 transformation
                    if transformation_name == 'mr2':
                        # for each item
                        for item_name, d in effect.items():
                            # apply the transformations
                            # do fill_color first
                            for color, color_effect in d['fill_color'].items():
                                # skip if transformation is unsafe
                                if color_effect == 'unsafe':
                                    continue

                                # do and save transformation
                                # result = mask_objects(image, item_name, color)
                                save_path = root_out_dir / f'{img_Path.stem}_{item_name}_{color}{img_Path.suffix}'
                                # result.save(save_path)

                                # figure out correct answer
                                if color_effect == 'invariant':
                                    # leave correct answer as blank, to be filled in later?
                                    correct_answer = ''
                                else:
                                    # pass through the mapping, "create" correct answer later?
                                    correct_answer = color_effect['equivariant']

                                # create unit test
                                unit_test_dict = {
                                    'description': f'{item_name} turned {color}',
                                    'image_paths': [str(save_path)],
                                    'answer': correct_answer
                                }

                                # add unit test to list, do be saved to json later
                                if question_id in res:
                                    res[question_id].append(unit_test_dict)
                                else:
                                    res[question_id] = [unit_test_dict]

                            # then do occlude bounding box
                            if d['occlude_bbox'] != 'unsafe':
                                bbox_effect = d['occlude_bbox']
                                # do and save transformation
                                # result = something
                                save_path = root_out_dir / f'{img_Path.stem}_{item_name}_occlude_bbox{img_Path.suffix}'
                                # result.save(save_path)
                            
                                # figure out correct answer
                                if bbox_effect == 'invariant':
                                    # leave correct answer as blank, to be filled in later?
                                    correct_answer = ''
                                else:
                                    # pass through the mapping, "create" correct answer later?
                                    correct_answer = bbox_effect['equivariant']

                                # create unit test dict
                                unit_test_dict = {
                                    'description': f'{item_name}, occluded bounding box',
                                    'image_paths': [str(save_path)],
                                    'answer': correct_answer
                                }
                                # add unit test to list, do be saved to json later
                                if question_id in res:
                                    res[question_id].append(unit_test_dict)
                                else:
                                    res[question_id] = [unit_test_dict]


                    # mr1 transformation
                    else:
                        # apply transformation
                        if transformation_name == 'hflip':
                            # result = flip(image, 'left')
                            pass
                        elif transformation_name == 'vflip':
                            # result = flip(image, 'up')
                            pass
                        elif transformation_name == 'grayscale':
                            # result = color_filter(image, 'grayscale')
                            pass
                        # save transformed image
                        save_path = root_out_dir / f'{img_Path.stem}_{transformation_name}{img_Path.suffix}'
                        # result.save(save_path)

                        # figure out correct answer
                        if effect == 'invariant':
                            # leave correct answer as blank, to be filled in later?
                            correct_answer = ''
                        else:
                            # pass through the mapping, "create" correct answer later?
                            correct_answer = effect['equivariant']

                        # create unit test dict
                        unit_test_dict = {
                            'description': transformation_name,
                            'image_paths': [str(save_path)],
                            'answer': correct_answer
                        }
                        # add unit test to list, do be saved to json later
                        if question_id in res:
                            res[question_id].append(unit_test_dict)
                        else:
                            res[question_id] = [unit_test_dict]
        with open(out_json, "w", encoding='utf-8') as f:
            json.dump(res, f, indent=2)
        return res
    except Exception:
        traceback.print_exc()
        raise


def apply_transformations(j: str, img: str, out: str, out_json: str, *,
                          num_tests: int = 2, seed: int = 0) -> dict:
    """Materialize predictions for a source image, keyed by question ID."""
    predictions = json.loads(j)
    # Preserve the new branch's [question_id, effects] manifest format alongside
    # the existing materialized {question_id, parsed} format used by our tests.
    if predictions and isinstance(predictions[0], list):
        return create_test_manifest(j, img, out, out_json)
    result = {}
    for prediction in predictions:
        question_id = str(prediction['question_id'])
        # Hash IDs so arbitrary dataset IDs cannot escape the output directory.
        directory = Path(out) / hashlib.sha256(question_id.encode()).hexdigest()
        result[question_id] = materialize_tests(
            img, eligible_transformations(prediction['parsed']), directory,
            num_tests=num_tests, seed=seed,
        )
    destination = Path(out_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('image', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('output_json', type=Path)
    parser.add_argument('--num-tests', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    apply_transformations(args.predictions.read_text(encoding='utf-8'), args.image,
                          args.output_dir, args.output_json,
                          num_tests=args.num_tests, seed=args.seed)
