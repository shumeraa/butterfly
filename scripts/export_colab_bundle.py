"""Bundle the transform package for Colab upload.

Run from the repository root: python -m scripts.export_colab_bundle
"""
from pathlib import Path
from zipfile import ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'outputs' / 'butterfly-transforms-colab.zip'
    output.parent.mkdir(exist_ok=True)
    sources = sorted((root / 'image_transforms').glob('*.py'))
    sources += [root / 'requirements-transforms.txt', root / 'requirements-masking.txt']
    with ZipFile(output, 'w') as archive:
        for source in sources:
            archive.write(source, source.relative_to(root))
    print(output)


if __name__ == '__main__':
    main()
