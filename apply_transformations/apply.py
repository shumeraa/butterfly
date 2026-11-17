'''
Takes in json output from LLM and runs the specified transformations on the image

TODO: need to hand over transformed image and expected answer to next step, what is the format that the next step expects?
Also, need to implement other transforms, but currently do not have the llm output to support that
will need to implement masking when we get the llm output for that
Need "correct" answers, different for test and for inference

what i did:
 - wrote code to add our custom unit tests to the ones created by Viunit
 - no need to mess with sampling part
 - just need clarification on llm output specifics
'''

import json
import sys
from pathlib import Path

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
def apply_transformations(j: str, img: str, out: str, out_json: str) -> dict:
    try:
        obj = json.loads(j)
        root_out_dir = Path(out)
        img_Path = Path(img)
        res = dict() # questionID: list of tests
        for q in obj:
            # figure out which transforms are valid
            parsed = q["parsed"]
            image_id = q['image_id']
            question_id = q['question_id']
            valid_transformations = []
            for transformation, d in parsed.items():
                if d["relation"] != "unsafe":
                    valid_transformations.append(transformation)

            subset = random.sample(valid_transformations, k=2)
            with Image.open(img) as source:
                image = ImageOps.exif_transpose(source)
                # if args.transform == 'apply':
                #     result = apply_transform(image, args.name, seed=args.seed)
                for transformation in subset:
                    result = apply_transform(image, transformation)

                    save_path = root_out_dir / f'{img_Path.stem}_{transformation}{img_Path.suffix}'
                    result.save(save_path)

                    d = {
                        'description': transformation,
                        'image_paths': [str(save_path)],
                        'answer': None
                    }
                    if question_id in res:
                        res[question_id].append(d)
                    else:
                        res[question_id] = [d]
        with open(out_json, "w", encoding='utf-8') as f:
            json.dump(res, f, indent=2)
        return res
    except Exception as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    json_text = Path("C:/Users/jnche/Downloads/predictions.json").read_text(encoding="utf-8")
    apply_transformations(json_text, "./test/1.png", "./outputDir")
