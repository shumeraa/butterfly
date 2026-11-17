"""Save a labeled contact sheet; every variant starts from the same source.

Run from the repository root:
    python -m scripts.preview_transforms 'assets/ViUnit Demo.png' outputs/transform_preview.png
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from image_transforms import COLOR_PRESETS, blur, color_filter, flip, hsv, noise, rotate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with Image.open(args.input) as image:
        source = ImageOps.exif_transpose(image).convert('RGB')
    source.thumbnail((480, 320))
    variants = [('Original', source), ('Flip up', flip(source, 'up')),
                ('Flip left', flip(source, 'left')), ('Rotate +30', rotate(source, 30)),
                ('Rotate -30', rotate(source, -30))]
    variants.extend((name.title(), color_filter(source, name)) for name in COLOR_PRESETS)
    variants.extend([('Blur radius=2', blur(source, 2)),
                     ('Noise std=.05, seed=42', noise(source, .05, seed=42)),
                     ('HSV (+30, -.2, +.1)', hsv(source, 30, -.2, .1))])
    width, height, columns = 500, 360, 3
    sheet = Image.new('RGB', (width * columns, height * ((len(variants) + columns - 1) // columns)), '#dddddd')
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(variants):
        image.thumbnail((width - 20, height - 40))
        x, y = index % columns * width, index // columns * height
        draw.text((x + 10, y + 8), label, fill='black')
        sheet.paste(image, (x + (width - image.width) // 2, y + 32))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == '__main__':
    main()
