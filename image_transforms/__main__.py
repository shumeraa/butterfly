"""Apply an image transformation and save the result."""
import argparse
from pathlib import Path

from PIL import Image, ImageOps

from . import COLOR_PRESETS, blur, color_filter, flip, hsv, noise, rotate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path, help='Use PNG to preserve alpha and exact pixels')
    commands = parser.add_subparsers(dest='transform', required=True)
    commands.add_parser('flip').add_argument('direction', choices=('up', 'left'))
    rotation = commands.add_parser('rotate')
    rotation.add_argument('angle', type=float)
    rotation.add_argument('--no-expand', action='store_true')
    commands.add_parser('color').add_argument('which', choices=COLOR_PRESETS)
    commands.add_parser('blur').add_argument('radius', type=float)
    noise_parser = commands.add_parser('noise')
    noise_parser.add_argument('amount', type=float)
    noise_parser.add_argument('--seed', type=int)
    hsv_parser = commands.add_parser('hsv')
    for channel in ('h', 's', 'v'):
        hsv_parser.add_argument(f'--delta-{channel}', type=float, default=0)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error('input and output must be different files')
    try:
        with Image.open(args.input) as source:
            image = ImageOps.exif_transpose(source)
            if args.transform == 'flip':
                result = flip(image, args.direction)
            elif args.transform == 'rotate':
                result = rotate(image, args.angle, expand=not args.no_expand)
            elif args.transform == 'color':
                result = color_filter(image, args.which)
            elif args.transform == 'blur':
                result = blur(image, args.radius)
            elif args.transform == 'noise':
                result = noise(image, args.amount, seed=args.seed)
            else:
                result = hsv(image, args.delta_h, args.delta_s, args.delta_v)
        result.save(args.output)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
