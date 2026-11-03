"""Contract tests for the six filters in the GQA reliability prompt."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import image_transforms as api
from image_transforms import high_level as high
from image_transforms import transforms as low

ROOT = Path(__file__).resolve().parents[1]
# Independent contract from origin/llm_reliability_experiment, not the registry.
PROMPT_KEYS = ('horizontal_flip', 'vertical_flip', 'rotate_180', 'grayscale', 'blur', 'noise')


class HighLevelTransformTests(unittest.TestCase):
    def setUp(self):
        self.pixels = np.random.default_rng(17).integers(0, 256, (9, 11, 3), dtype=np.uint8)
        self.image = Image.fromarray(self.pixels)

    def assertPixels(self, actual, expected):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_exact_prompt_keys_and_order(self):
        self.assertEqual(high.TRANSFORMS, PROMPT_KEYS)
        self.assertEqual(api.HIGH_LEVEL_TRANSFORMS, PROMPT_KEYS)
        self.assertEqual(tuple(api.transform_images(self.image, seed=42)), PROMPT_KEYS)

    def test_dispatch_matches_low_level_operations(self):
        expected = {
            'horizontal_flip': low.flip(self.image, 'left'),
            'vertical_flip': low.flip(self.image, 'up'),
            'rotate_180': low.rotate(self.image, 180),
            'grayscale': low.color_filter(self.image, 'grayscale'),
            'blur': low.blur(self.image, 2),
            'noise': low.noise(self.image, .05, seed=42),
        }
        for name, result in expected.items():
            with self.subTest(name=name):
                self.assertPixels(api.apply_transform(self.image, name, seed=42), result)

    def test_geometry_known_arrays(self):
        for name, expected in [('horizontal_flip', self.pixels[:, ::-1]),
                               ('vertical_flip', self.pixels[::-1]),
                               ('rotate_180', np.rot90(self.pixels, 2))]:
            self.assertPixels(api.apply_transform(self.image, name), expected)

    def test_named_wrappers_and_fixed_parameters(self):
        for name, function, args, kwargs in [
            ('horizontal_flip', 'flip', ('left',), {}),
            ('vertical_flip', 'flip', ('up',), {}),
            ('rotate_180', 'rotate', (180,), {}),
            ('grayscale', 'color_filter', ('grayscale',), {}),
            ('blur', 'blur', (2,), {}),
            ('noise', 'noise', (.05,), {'seed': 123}),
        ]:
            with self.subTest(name=name), patch.object(low, function, return_value=self.image) as operation:
                result = getattr(high, name)(self.image, **({'seed': 123} if name == 'noise' else {}))
                self.assertIs(result, self.image)
                operation.assert_called_once_with(self.image, *args, **kwargs)

    def test_grayscale_is_luminance_not_hsv_desaturation(self):
        image = Image.new('RGB', (1, 1), (100, 150, 200))
        self.assertEqual(high.grayscale(image).getpixel((0, 0)), (141, 141, 141))

    def test_batch_variants_start_from_original(self):
        before = self.image.tobytes()
        variants = api.transform_images(self.image, 789)
        for name, result in variants.items():
            with self.subTest(name=name):
                self.assertPixels(result, api.apply_transform(self.image, name, seed=789))
                self.assertIsNot(result, self.image)
                self.assertEqual(result.size, self.image.size)
        self.assertEqual(len({id(result) for result in variants.values()}), 6)
        others = {name: result.tobytes() for name, result in variants.items() if name != 'horizontal_flip'}
        variants['horizontal_flip'].paste('white', (0, 0, self.image.width, self.image.height))
        self.assertEqual(self.image.tobytes(), before)
        for name, pixels in others.items():
            self.assertEqual(variants[name].tobytes(), pixels)

    def test_noise_seed_repeatability_and_variation(self):
        image = Image.new('RGB', (128, 128), (128, 128, 128))
        first = high.noise(image, seed=42)
        self.assertPixels(first, high.noise(image, seed=42))
        self.assertPixels(first, api.apply_transform(image, 'noise', seed=42))
        self.assertPixels(first, api.transform_images(image, seed=42)['noise'])
        self.assertFalse(np.array_equal(first, high.noise(image, seed=43)))
        difference = np.asarray(first, dtype=float) - 128
        self.assertAlmostEqual(difference.mean(), 0, delta=.15)
        self.assertAlmostEqual(difference.std(), 12.75, delta=.15)

    def test_invalid_noise_seeds(self):
        for seed in [-1, True, .5, '42', float('nan')]:
            for call in [lambda: high.noise(self.image, seed=seed),
                         lambda: api.apply_transform(self.image, 'noise', seed=seed),
                         lambda: api.transform_images(self.image, seed=seed)]:
                with self.subTest(seed=seed), self.assertRaises(ValueError):
                    call()

    def test_alpha_preserved_or_transformed(self):
        image = self.image.convert('RGBA')
        alpha = np.arange(99, dtype=np.uint8).reshape(9, 11)
        image.putalpha(Image.fromarray(alpha))
        before = image.tobytes()
        for name, result in api.transform_images(image, 5).items():
            expected_alpha = {'horizontal_flip': alpha[:, ::-1], 'vertical_flip': alpha[::-1],
                              'rotate_180': np.rot90(alpha, 2)}.get(name, alpha)
            with self.subTest(name=name):
                self.assertEqual(result.mode, 'RGBA')
                self.assertPixels(result.getchannel('A'), expected_alpha)
        self.assertEqual(image.tobytes(), before)

    def test_mode_normalization(self):
        images = [self.image.convert(mode) for mode in ('L', 'LA', 'CMYK')]
        palette = self.image.quantize()
        palette.info['transparency'] = 0
        images.append(palette)
        for image, expected_mode in zip(images, ('RGB', 'RGBA', 'RGB', 'RGBA')):
            for name, result in api.transform_images(image, 3).items():
                with self.subTest(mode=image.mode, name=name):
                    self.assertEqual(result.mode, expected_mode)
                    self.assertEqual(result.size, image.size)

    def test_strict_names_no_response_parsing(self):
        for name in ['', 'flip', 'Horizontal_flip', 'horizontal flip', None, True, 1, [], {},
                     '{"noise": {"relation": "same", "reason": "test"}}']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                api.apply_transform(self.image, name)

    def test_invalid_images(self):
        for name in PROMPT_KEYS:
            with self.subTest(name=name), self.assertRaises(TypeError):
                api.apply_transform(self.pixels, name, seed=42)

    def test_low_level_api_is_unchanged(self):
        self.assertIs(api.blur, low.blur)
        self.assertIs(api.noise, low.noise)
        self.assertPixels(api.blur(self.image, 0), self.image)
        self.assertPixels(api.noise(self.image, 0, seed=1), self.image)

    def test_import_does_not_load_ml_dependencies(self):
        run = subprocess.run([sys.executable, '-c',
                              'import sys; from image_transforms import high_level, transform_images; '
                              'assert "torch" not in sys.modules; assert "transformers" not in sys.modules'],
                             capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_manual_cli_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / 'source.png', Path(directory) / 'result.png'
            self.image.save(source)
            run = subprocess.run([sys.executable, '-m', 'image_transforms', str(source), str(dest),
                                  'apply', 'noise', '--seed', '42'],
                                 capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(run.returncode, 0, run.stderr)
            with Image.open(dest) as result:
                self.assertPixels(result, high.noise(self.image, seed=42))


if __name__ == '__main__':
    unittest.main()
