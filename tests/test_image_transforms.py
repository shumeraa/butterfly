import colorsys
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import image_transforms as f

ROOT = Path(__file__).resolve().parents[1]


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.pixels = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                                [[255, 255, 0], [0, 255, 255], [255, 0, 255]]], dtype=np.uint8)
        self.image = Image.fromarray(self.pixels)

    def assertPixels(self, image, expected):
        np.testing.assert_array_equal(np.asarray(image), expected)

    def test_flips_and_inverse(self):
        for direction, expected in [('up', self.pixels[::-1]), ('left', self.pixels[:, ::-1])]:
            self.assertPixels(f.flip(self.image, direction), expected)
            self.assertPixels(f.flip(f.flip(self.image, direction), direction), self.pixels)

    def test_rotations(self):
        for angle, turns in [(90, 1), (-90, -1), (180, 2), (450, 1)]:
            self.assertPixels(f.rotate(self.image, angle), np.rot90(self.pixels, turns))
        result = f.rotate(Image.new('RGB', (20, 10), 'white'), 45)
        self.assertGreater(result.width, 20)
        self.assertGreater(result.height, 10)
        self.assertEqual(f.rotate(self.image, 45, expand=False).size, self.image.size)

    def test_presets_known_pixel(self):
        image = Image.new('RGB', (1, 1), (100, 150, 200))
        expected = {'grayscale': (141, 141, 141), 'invert': (155, 105, 55),
                    'sepia': (192, 171, 134), 'warm': (120, 150, 180), 'cool': (80, 150, 220)}
        for name, pixel in expected.items():
            self.assertEqual(f.color_filter(image, name).getpixel((0, 0)), pixel)
        self.assertEqual(f.color_filter(Image.new('RGB', (1, 1), 'white'), 'sepia').getpixel((0, 0)),
                         (255, 255, 239))

    def test_identity(self):
        for result in [f.rotate(self.image, 0), f.blur(self.image, 0),
                       f.noise(self.image, 0), f.hsv(self.image), f.hsv(self.image, 360)]:
            self.assertPixels(result, self.pixels)
            self.assertIsNot(result, self.image)

    def test_blur_softens_impulse(self):
        image = Image.new('RGB', (21, 21))
        image.putpixel((10, 10), (255, 255, 255))
        blurred = f.blur(image, 2)
        self.assertTrue(0 < blurred.getpixel((10, 10))[0] < 255)
        self.assertGreater(blurred.getpixel((9, 10))[0], 0)

    def test_noise_distribution_seed_and_clipping(self):
        image = Image.new('RGB', (300, 300), (128, 128, 128))
        first = f.noise(image, .05, seed=42)
        self.assertPixels(first, np.asarray(f.noise(image, .05, seed=42)))
        self.assertFalse(np.array_equal(first, f.noise(image, .05, seed=43)))
        difference = np.asarray(first, dtype=float) - 128
        self.assertAlmostEqual(difference.mean(), 0, delta=.1)
        self.assertAlmostEqual(difference.std(), .05 * 255, delta=.1)
        extremes = np.asarray(f.noise(image, 1, seed=42))
        self.assertEqual(extremes.min(), 0)
        self.assertEqual(extremes.max(), 255)

    def test_hsv_matches_colorsys(self):
        pixels = np.random.default_rng(6).integers(0, 256, (15, 20, 3), dtype=np.uint8)
        pixels[0, :3] = [[0, 0, 0], [128, 128, 128], [255, 255, 255]]
        for dh, ds, dv in [(120, 0, 0), (-480, .2, -.3), (725, -.5, .4)]:
            expected = []
            for pixel in pixels.reshape(-1, 3):
                h, s, v = colorsys.rgb_to_hsv(*(pixel / 255))
                expected.append(colorsys.hsv_to_rgb((h + dh / 360) % 1,
                                np.clip(s + ds, 0, 1), np.clip(v + dv, 0, 1)))
            expected = np.rint(np.array(expected).reshape(pixels.shape) * 255).astype(np.uint8)
            np.testing.assert_allclose(f.hsv(Image.fromarray(pixels), dh, ds, dv), expected, atol=1)
        self.assertPixels(f.hsv(self.image, 0, -1, 1), np.full_like(self.pixels, 255))
        self.assertPixels(f.hsv(self.image, 0, 0, -1), np.zeros_like(self.pixels))

    def test_alpha_modes_and_source_unchanged(self):
        image = self.image.convert('RGBA')
        image.putalpha(Image.fromarray(np.array([[0, 20, 40], [80, 160, 255]], dtype=np.uint8)))
        before = image.tobytes()
        operations = [lambda im: f.blur(im, 1), lambda im: f.noise(im, .2, seed=1),
                      lambda im: f.hsv(im, 45, .1, -.1)]
        operations += [lambda im, name=name: f.color_filter(im, name) for name in f.COLOR_PRESETS]
        for operation in operations:
            self.assertPixels(operation(image).getchannel('A'), np.asarray(image.getchannel('A')))
            self.assertEqual(image.tobytes(), before)
        self.assertPixels(f.flip(image, 'up'), np.asarray(image)[::-1])
        self.assertPixels(f.rotate(image, 90), np.rot90(np.asarray(image)))
        self.assertEqual(f.hsv(Image.new('L', (2, 2))).mode, 'RGB')
        palette = Image.new('P', (2, 2))
        palette.info['transparency'] = 0
        self.assertEqual(f.color_filter(palette, 'invert').getpixel((0, 0))[3], 0)
        self.assertPixels(self.image, self.pixels)

    def test_invalid_parameters(self):
        cases = [
            (f.flip, ('right',), {}),
            (f.color_filter, ('unknown',), {}),
            (f.rotate, ('90',), {}),
            (f.rotate, (10,), {'expand': 1}),
            (f.blur, (-1,), {}),
            (f.noise, (1.1,), {}),
            (f.noise, (.1,), {'seed': -1}),
            (f.noise, (.1,), {'seed': 1.5}),
            (f.hsv, (), {'delta_s': -2}),
            (f.hsv, (), {'delta_v': 2}),
        ]
        for value in [float('nan'), float('inf'), True]:
            cases.extend([(f.blur, (value,), {}), (f.hsv, (), {'delta_h': value})])
        for function, args, kwargs in cases:
            with self.subTest(transform=function.__name__, args=args, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    function(self.image, *args, **kwargs)
        with self.assertRaises(TypeError):
            f.flip(np.zeros((2, 2, 3)), 'up')

    def test_cli_all_transforms_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / 'source.png', Path(directory) / 'result.png'
            self.image.save(source)
            cases = [(['flip', 'left'], f.flip(self.image, 'left')),
                     (['rotate', '-90'], f.rotate(self.image, -90)),
                     (['color', 'sepia'], f.color_filter(self.image, 'sepia')),
                     (['blur', '1'], f.blur(self.image, 1)),
                     (['noise', '.1', '--seed', '3'], f.noise(self.image, .1, seed=3)),
                     (['hsv', '--delta-h', '90', '--delta-v', '-.1'], f.hsv(self.image, 90, 0, -.1))]
            for arguments, expected in cases:
                run = subprocess.run([sys.executable, '-m', 'image_transforms', str(source),
                                      str(dest), *arguments], capture_output=True, text=True, cwd=ROOT)
                self.assertEqual(run.returncode, 0, run.stderr)
                with Image.open(dest) as result:
                    self.assertPixels(result, np.asarray(expected))
            before = source.read_bytes()
            for out, arguments in [(source, ['flip', 'up']), (dest, ['blur', '-1'])]:
                run = subprocess.run([sys.executable, '-m', 'image_transforms', str(source),
                                      str(out), *arguments], capture_output=True, text=True, cwd=ROOT)
                self.assertNotEqual(run.returncode, 0)
                self.assertNotIn('Traceback', run.stderr)
            self.assertEqual(source.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
