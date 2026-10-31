import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from image_transforms import GroundedSAM, mask_objects


class MaskingTests(unittest.TestCase):
    def setUp(self):
        self.pixels = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
        self.image = Image.fromarray(self.pixels)
        self.mask = np.zeros((4, 5), dtype=bool)
        self.mask[0, :2] = True
        self.mask[3, 3:] = True
        self.segmenter = Mock()
        self.segmenter.segment.return_value = self.mask

    def test_white_and_black_fill(self):
        for fill, value in [('white', 255), ('black', 0)]:
            result = mask_objects(self.image, 'person', fill, segmenter=self.segmenter)
            pixels = np.asarray(result)
            self.assertTrue(np.all(pixels[self.mask] == value))
            np.testing.assert_array_equal(pixels[~self.mask], self.pixels[~self.mask])
        self.assertEqual(self.segmenter.segment.call_args.args[1], 'person')
        np.testing.assert_array_equal(self.image, self.pixels)

    def test_no_matches_returns_copy(self):
        self.segmenter.segment.return_value = np.zeros_like(self.mask)
        result = mask_objects(self.image, 'car', segmenter=self.segmenter)
        self.assertIsNot(result, self.image)
        np.testing.assert_array_equal(result, self.image)

    def test_alpha_is_preserved(self):
        image = self.image.convert('RGBA')
        image.putalpha(Image.fromarray(np.arange(20, dtype=np.uint8).reshape(4, 5)))
        before = image.tobytes()
        result = mask_objects(image, 'person', segmenter=self.segmenter)
        np.testing.assert_array_equal(result.getchannel('A'), image.getchannel('A'))
        self.assertEqual(image.tobytes(), before)
        self.assertEqual(self.segmenter.segment.call_args.args[0].mode, 'RGB')

    def test_default_backend_is_used(self):
        with patch('image_transforms.masking._default_segmenter', return_value=self.segmenter):
            result = mask_objects(self.image, 'person')
        self.assertTrue(np.all(np.asarray(result)[self.mask] == 255))

    def test_invalid_parameters_do_not_load_models(self):
        with patch('image_transforms.masking._default_segmenter') as backend:
            for word in ['', ' ', '.', None, 12]:
                with self.assertRaises(ValueError):
                    mask_objects(self.image, word)
            with self.assertRaises(ValueError):
                mask_objects(self.image, 'person', fill='red')
            backend.assert_not_called()
        with self.assertRaises(ValueError):
            GroundedSAM(box_threshold=1.1)

    def test_invalid_backend_mask(self):
        for mask in [np.zeros((5, 4), dtype=bool), np.zeros((4, 5), dtype=np.uint8)]:
            self.segmenter.segment.return_value = mask
            with self.assertRaises(ValueError):
                mask_objects(self.image, 'person', segmenter=self.segmenter)

    def test_import_does_not_require_torch_or_transformers(self):
        run = subprocess.run([sys.executable, '-c',
                              'import sys; import image_transforms; '
                              'assert "torch" not in sys.modules; '
                              'assert "transformers" not in sys.modules'],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
