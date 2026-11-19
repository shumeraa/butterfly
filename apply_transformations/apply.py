'''
Takes in json output from LLM and runs the specified transformations on the image

TODO: 
Need "correct" answers, different for test and for inference

next steps:
1. test apply_transformations function (done, generates a correct test_out.json file, using dummy input values for now)
2. look at how to turn correct answers into actual correct answers
3. see if we can run generate.py

don't have occlude bbox code yet?
'''

import json
import sys
from pathlib import Path
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageOps
from image_transforms import (HIGH_LEVEL_TRANSFORMS, GroundedSAM, apply_transform, mask_objects,
               COLOR_PRESETS, blur, color_filter, flip, hsv, noise, rotate)
import random

def append_custom_unit_tests(dataset_data: list, tests_by_question_id: dict) -> int:
    """Append custom tests in place after ViUniT selects tests and gets images.

    Requires unit_test_generation.use_program=False. Mapping keys identify
    questionId (or sample_id if questionId is absent), never imageId. Values:
    [{"description": "Flipped car", "answer": "red",
      "image_paths": ["tests/car_flipped.jpg"]}]

    Every supplied case is selected. Exact description/answer/path duplicates
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
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Each custom test needs a known expected answer")
            if not isinstance(paths, list) or not paths:
                raise ValueError("image_paths must be a nonempty list")
            if any(not isinstance(p, (str, Path)) or not str(p) for p in paths):
                raise ValueError("Image paths must be nonempty strings or Paths")
            pending.append((entry, (description, answer), [str(p) for p in paths]))

    added = 0
    for entry, test, paths in pending:
        already_present = any(
            tuple(existing_test) == test and existing_images == paths
            for existing_test, existing_images in zip(
                entry["selected_unit_tests"], entry["unit_test_images"]
            )
        )
        if not already_present:
            entry["selected_unit_tests"].append(test)
            entry["unit_test_images"].append(paths)
            added += 1
    return added

# still needs a bit of fixing
def apply_transformations(json_string: str, img: str, out: str, out_json: str) -> dict:
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
        print(color_effect)
        exit(1)


if __name__ == '__main__':
    json_text = Path("C:/Users/jnche/Downloads/sample_llm_output.json").read_text(encoding="utf-8")
    apply_transformations(json_text, "C:/Users/jnche/Pictures/Screenshots/Screenshot 2026-11-15 213810.png", "./outputDir", './test_out.json')
