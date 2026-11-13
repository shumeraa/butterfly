'''
Takes in json output from LLM and runs the specified transformations on the image

TODO: need to hand over transformed image and expected answer to next step, what is the format that the next step expects?
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

def apply_transformations(j: str, img: str, out: str) -> None:
    try:
        obj = json.loads(j)
        root_out_dir = Path(out)
        img_Path = Path(img)
        for q in obj:
            # figure out which transforms are valid
            parsed = q["parsed"]
            image_id = q['image_id']
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
                    if transformation == 'horizontal_flip':
                        result = flip(image, 'left')
                    elif transformation == 'vertical_flip':
                        result = flip(image, 'up')
                    elif transformation == 'rotate_180':
                        result = rotate(image, 180)
                    elif transformation == 'grayscale':
                        result = color_filter(image, 'grayscale')
                    elif transformation == 'blur':
                        result = blur(image, 2)
                    elif transformation == 'noise':
                        result = noise(image, 2)
                    else:
                        raise Exception(f'{transformation} is not valid.')

                    save_path = root_out_dir / f'{img_Path.stem}_{transformation}{img_Path.suffix}'
                    result.save(save_path)
                    '''
                    elif args.transform == 'mask':
                        segmenter = GroundedSAM(device=args.device)
                        result = mask_objects(image, args.word, args.fill, segmenter=segmenter)
                    else:
                        result = hsv(image, args.delta_h, args.delta_s, args.delta_v)
                    '''
    except Exception as e:
        print(e)


if __name__ == '__main__':
    json_text = Path("C:/Users/jnche/Downloads/predictions.json").read_text(encoding="utf-8")
    apply_transformations(json_text, "./test/1.png", "./outputDir")
