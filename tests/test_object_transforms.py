"""CPU-only contracts for reusable segmentation/transform composition."""
from functools import partial
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import image_transforms as api
from image_transforms import high_level as high
from image_transforms.objects import ANCHORS


class ScaleTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(np.random.default_rng(3).integers(0, 256, (6, 8, 4), dtype=np.uint8))

    def test_uniform_and_independent_factors(self):
        for sx, sy, size in [(2, None, (16, 12)), (.5, None, (4, 3)),
                             (1, 1.5, (8, 9)), (2, .5, (16, 3)),
                             (.01, .01, (1, 1))]:
            with self.subTest(sx=sx, sy=sy):
                result = api.scale(self.image, sx, sy)
                self.assertEqual(result.size, size)
                self.assertEqual(result.mode, 'RGBA')
                np.testing.assert_array_equal(result, self.image.resize(size, Image.Resampling.BICUBIC))

    def test_identity_and_source_preserved(self):
        before = self.image.tobytes()
        result = api.scale(self.image)
        self.assertIsNot(result, self.image)
        self.assertEqual(result.tobytes(), before)
        api.scale(self.image, 2, .5)
        self.assertEqual(self.image.tobytes(), before)

    def test_modes(self):
        for mode, expected in [('L', 'RGB'), ('LA', 'RGBA'), ('RGB', 'RGB')]:
            self.assertEqual(api.scale(self.image.convert(mode), 2).mode, expected)
        palette = self.image.convert('RGB').quantize()
        palette.info['transparency'] = 0
        self.assertEqual(api.scale(palette, 2).mode, 'RGBA')

    def test_invalid_factors(self):
        for value in [0, -1, True, '2', float('nan'), float('inf'), 1e308]:
            for sx, sy in [(value, 1), (1, value)]:
                with self.subTest(sx=sx, sy=sy), self.assertRaises(ValueError):
                    api.scale(self.image, sx, sy)
        with self.assertRaises(TypeError):
            api.scale(np.zeros((2, 2, 3)), 2)

    def test_scale_is_not_exposed_in_mr1_registry(self):
        self.assertNotIn('scale', api.HIGH_LEVEL_TRANSFORMS)
        with self.assertRaises(ValueError):
            api.apply_transform(self.image, 'scale')


class MaskedAppearanceTests(unittest.TestCase):
    def setUp(self):
        self.pixels = np.random.default_rng(19).integers(0, 256, (12, 16, 4), dtype=np.uint8)
        self.image = Image.fromarray(self.pixels)
        self.mask = np.zeros((12, 16), dtype=bool)
        self.mask[2:10, 4:12] = True
        self.mask[4:6, 6:8] = False

    def test_reuse_existing_appearance_filters(self):
        operations = [(high.grayscale, {}), (high.blur, {}), (high.noise, {'seed': 42}),
                      (api.hsv, {'delta_h': 35, 'delta_s': -.2, 'delta_v': .1}),
                      (api.color_filter, {'which': 'warm'})]
        for transform, kwargs in operations:
            with self.subTest(transform=transform.__name__):
                function = Mock(wraps=transform)
                result = api.transform_masked(self.image, self.mask, function, **kwargs)
                function.assert_called_once()
                self.assertEqual(function.call_args.kwargs, kwargs)
                pixels = np.array(result)
                expected = np.array(transform(self.image, **kwargs))
                np.testing.assert_array_equal(pixels[self.mask], expected[self.mask])
                np.testing.assert_array_equal(pixels[~self.mask], self.pixels[~self.mask])
                np.testing.assert_array_equal(pixels[..., 3], self.pixels[..., 3])
                np.testing.assert_array_equal(self.image, self.pixels)
                self.assertEqual(result.mode, 'RGBA')

    def test_empty_mask_does_not_call_transform(self):
        function = Mock()
        result = api.transform_masked(self.image, np.zeros_like(self.mask), function)
        function.assert_not_called()
        self.assertIsNot(result, self.image)
        np.testing.assert_array_equal(result, self.pixels)

    def test_input_modes_and_mutating_callable(self):
        def mutate(image):
            image.paste('white', (0, 0, image.width, image.height))
            return image
        source = self.image.convert('RGB')
        before = source.tobytes()
        result = api.transform_masked(source, self.mask, mutate)
        self.assertEqual(source.tobytes(), before)
        self.assertTrue(np.all(np.array(result)[self.mask] == 255))
        self.assertEqual(api.transform_masked(self.image.convert('L'), self.mask, high.grayscale).mode, 'RGB')

    def test_masks_are_reusable_and_not_modified(self):
        before = self.mask.copy()
        self.mask.setflags(write=False)
        for transform in (high.grayscale, high.blur):
            api.transform_masked(self.image, self.mask, transform)
        np.testing.assert_array_equal(self.mask, before)

    def test_invalid_masks(self):
        for mask in [self.mask.astype(np.uint8), self.mask.T, self.mask[..., None]]:
            with self.subTest(shape=mask.shape), self.assertRaises(ValueError):
                api.transform_masked(self.image, mask, high.blur)

    def test_invalid_transform_outputs(self):
        transforms = [lambda im: im.resize((1, 1)), lambda im: im.convert('RGB'),
                      lambda im: np.array(im)]
        for transform in transforms:
            with self.subTest(transform=transform), self.assertRaises((ValueError, TypeError)):
                api.transform_masked(self.image, self.mask, transform)
        def change_alpha(image):
            image.putalpha(255)
            return image
        with self.assertRaisesRegex(ValueError, 'preserve alpha'):
            api.transform_masked(self.image, self.mask, change_alpha)

    def test_invalid_options(self):
        options = [{'geometry': 1}, {'anchor': 'feet'}, {'anchor': []},
                   {'anchor': 'bottom_center'}, {'fill': 'white'},
                   {'allow_clipping': True}, {'allow_clipping': 1}]
        for kwargs in options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                api.transform_masked(self.image, self.mask, high.blur, **kwargs)
        with self.assertRaises(TypeError):
            api.transform_masked(self.image, self.mask, 'blur')


class ObjectGeometryTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (20, 20), 'white')
        self.box = (7, 8, 11, 14)
        self.patch = Image.fromarray(np.random.default_rng(5).integers(30, 225, (6, 4, 3), dtype=np.uint8))
        self.image.paste(self.patch, self.box[:2])
        self.mask = np.zeros((20, 20), dtype=bool)
        self.mask[8:14, 7:11] = True

    def test_known_geometry_cannot_accidentally_use_whole_scene_appearance_path(self):
        for transform in (api.flip, api.rotate, api.scale, high.horizontal_flip,
                          high.vertical_flip, high.rotate_180, partial(api.scale, scale_y=2)):
            with self.subTest(transform=transform), self.assertRaisesRegex(ValueError, 'geometry=True'):
                api.transform_masked(self.image, self.mask, transform)

    def test_taller_person_bottom_anchored(self):
        before = self.image.tobytes()
        function = Mock(wraps=api.scale)
        result = api.transform_masked(self.image, self.mask, function, geometry=True,
                                      scale_x=1, scale_y=1.5, anchor='bottom_center')
        function.assert_called_once()
        expected = Image.new('RGB', self.image.size, 'white')
        expected.paste(api.scale(self.patch, 1, 1.5), (7, 5))
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.size, self.image.size)
        self.assertEqual(self.image.tobytes(), before)

    def test_existing_geometric_filters_on_object_not_scene(self):
        for transform in (high.horizontal_flip, high.vertical_flip, high.rotate_180):
            with self.subTest(transform=transform.__name__):
                expected = self.image.copy()
                expected.paste(transform(self.patch), self.box[:2])
                result = api.transform_masked(self.image, self.mask, transform, geometry=True)
                np.testing.assert_array_equal(result, expected)

    def test_all_anchors_with_growth(self):
        for anchor, (ax, ay) in ANCHORS.items():
            with self.subTest(anchor=anchor):
                result = api.transform_masked(self.image, self.mask, api.scale, geometry=True,
                                              scale_x=2, scale_y=1.5, anchor=anchor)
                expected = Image.new('RGB', self.image.size, 'white')
                expected.paste(api.scale(self.patch, 2, 1.5),
                               (7 + round(-4 * ax), 8 + round(-3 * ay)))
                np.testing.assert_array_equal(result, expected)

    def test_shrink_requires_explicit_background(self):
        with self.assertRaisesRegex(ValueError, 'background or fill'):
            api.transform_masked(self.image, self.mask, api.scale, geometry=True, scale_x=.5)
        result = api.transform_masked(self.image, self.mask, api.scale, geometry=True,
                                      scale_x=.5, fill='magenta')
        expected = self.image.copy()
        expected.paste('magenta', self.box)
        expected.paste(api.scale(self.patch, .5), (8, 10))
        np.testing.assert_array_equal(result, expected)

    def test_background_applies_only_to_old_mask(self):
        background = Image.new('RGB', self.image.size, 'cyan')
        result = api.transform_masked(self.image, self.mask, api.scale, geometry=True,
                                      scale_x=.5, background=background)
        expected = self.image.copy()
        expected.paste('cyan', self.box)
        expected.paste(api.scale(self.patch, .5), (8, 10))
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(background.getpixel((0, 0)), (0, 255, 255))

    def test_arbitrary_rotation_reuses_primitive(self):
        result = api.transform_masked(self.image, self.mask, api.rotate, geometry=True,
                                      angle=90, fill='white')
        expected = Image.new('RGB', self.image.size, 'white')
        expected.paste(Image.fromarray(np.rot90(np.array(self.patch))), (6, 9))
        np.testing.assert_array_equal(result, expected)

    def test_irregular_mask_moves_without_bbox_background_or_ghost(self):
        mask = np.zeros_like(self.mask)
        mask[8, 7] = True
        mask[9, 7:10] = True
        image = Image.new('RGB', self.image.size, 'blue')
        pixels = np.array(image)
        pixels[mask] = (255, 0, 0)
        image = Image.fromarray(pixels)
        with self.assertRaisesRegex(ValueError, 'background or fill'):
            api.transform_masked(image, mask, high.horizontal_flip, geometry=True)
        result = api.transform_masked(image, mask, high.horizontal_flip, geometry=True, fill='blue')
        expected = np.full_like(pixels, (0, 0, 255))
        expected[8, 9] = (255, 0, 0)
        expected[9, 7:10] = (255, 0, 0)
        np.testing.assert_array_equal(result, expected)

    def test_isolated_rgba_layer_and_mask_passed_to_primitive(self):
        self.mask[9, 8] = False
        def inspect_layer(layer):
            self.assertEqual(layer.mode, 'RGBA')
            self.assertEqual(layer.size, (4, 6))
            self.assertEqual(layer.getpixel((1, 1)), (0, 0, 0, 0))
            np.testing.assert_array_equal(np.array(layer)[..., 3] > 0, self.mask[8:14, 7:11])
            return layer
        result = api.transform_masked(self.image, self.mask, inspect_layer, geometry=True)
        np.testing.assert_array_equal(result, self.image)

    def test_expanded_rotation_does_not_transport_background_colors(self):
        mask = np.zeros_like(self.mask)
        mask[8:14, 8] = True
        mask[13, 8:11] = True
        pixels = np.zeros((20, 20, 3), dtype=np.uint8)
        pixels[:] = (0, 255, 0)
        pixels[mask] = (255, 0, 0)
        image = Image.fromarray(pixels)
        result = api.transform_masked(image, mask, api.rotate, geometry=True,
                                      angle=25, fill='blue')
        changed = np.any(np.array(result) != pixels, axis=-1)
        # The old green bbox background is never carried into the object layer.
        self.assertTrue(np.all(np.array(result)[changed, 0] + np.array(result)[changed, 2] > 0))

    def test_clipping_requires_opt_in(self):
        image = Image.new('RGB', (8, 8), 'white')
        image.paste('red', (2, 1, 6, 5))
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:5, 2:6] = True
        with self.assertRaisesRegex(ValueError, 'leaves the canvas'):
            api.transform_masked(image, mask, api.scale, geometry=True, scale_x=1,
                                 scale_y=2, anchor='bottom_center')
        result = api.transform_masked(image, mask, api.scale, geometry=True, scale_x=1,
                                      scale_y=2, anchor='bottom_center', allow_clipping=True)
        self.assertEqual(result.size, (8, 8))
        expected = np.array(Image.new('RGB', (8, 8), 'white'))
        expected[:5, 2:6] = (255, 0, 0)
        np.testing.assert_array_equal(result, expected)

    def test_semitransparent_geometry_does_not_double_composite_old_object(self):
        image = self.image.convert('RGBA')
        alpha = np.full((20, 20), 255, dtype=np.uint8)
        alpha[self.mask] = 128
        image.putalpha(Image.fromarray(alpha))
        with self.assertRaisesRegex(ValueError, 'background or fill'):
            api.transform_masked(image, self.mask, high.horizontal_flip, geometry=True)
        result = api.transform_masked(image, self.mask, high.horizontal_flip,
                                      geometry=True, fill=(0, 0, 0, 0))
        expected = image.copy()
        expected.paste(high.horizontal_flip(image.crop(self.box)), self.box[:2])
        np.testing.assert_array_equal(result, expected)

    def test_identity_preserves_hidden_rgb_and_alpha(self):
        image = self.image.convert('RGBA')
        image.putalpha(0)
        result = api.transform_masked(image, self.mask, api.scale, geometry=True)
        self.assertIsNot(result, image)
        np.testing.assert_array_equal(result, image)

    def test_invalid_geometry_outputs_and_backgrounds(self):
        for transform in [lambda im: im.convert('RGB'), lambda im: None]:
            with self.subTest(transform=transform), self.assertRaises((ValueError, TypeError)):
                api.transform_masked(self.image, self.mask, transform, geometry=True)
        for kwargs in [{'fill': 'white', 'background': self.image},
                       {'background': Image.new('RGB', (1, 1))}, {'background': 'file.png'},
                       {'fill': 'not-a-color'}]:
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                api.transform_masked(self.image, self.mask, api.scale, geometry=True, **kwargs)


class ObjectSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (20, 20), (80, 120, 180))
        self.masks = np.zeros((2, 20, 20), dtype=bool)
        self.masks[0, 4:10, 2:6] = True
        self.masks[1, 8:14, 12:16] = True
        self.backend = Mock(spec=['segment', 'segment_instances'])
        self.backend.segment.return_value = self.masks.any(axis=0)
        self.backend.segment_instances.return_value = self.masks

    def test_union_appearance_preserves_existing_backend_interface(self):
        result = api.transform_objects(self.image, 'person', high.grayscale, segmenter=self.backend)
        expected = api.transform_masked(self.image, self.masks.any(axis=0), high.grayscale)
        np.testing.assert_array_equal(result, expected)
        self.backend.segment.assert_called_once()
        self.backend.segment_instances.assert_not_called()
        self.assertEqual(self.backend.segment.call_args.args[1], 'person')

    def test_explicit_instance_selects_only_one_object(self):
        result = api.transform_objects(self.image, 'person', api.hsv, segmenter=self.backend,
                                       instance=np.int64(1), delta_h=90)
        expected = api.transform_masked(self.image, self.masks[1], api.hsv, delta_h=90)
        np.testing.assert_array_equal(result, expected)
        self.backend.segment.assert_not_called()
        self.backend.segment_instances.assert_called_once()

    def test_geometry_requires_instance_when_multiple_match(self):
        with self.assertRaisesRegex(ValueError, 'select an instance'):
            api.transform_objects(self.image, 'person', api.scale, segmenter=self.backend,
                                  geometry=True, scale_y=1.5)
        result = api.transform_objects(self.image, 'person', api.scale, segmenter=self.backend,
                                       geometry=True, instance=1, scale_y=1.5, anchor='bottom_center')
        expected = api.transform_masked(self.image, self.masks[1], api.scale, geometry=True,
                                        scale_y=1.5, anchor='bottom_center')
        np.testing.assert_array_equal(result, expected)

    def test_single_geometry_instance_needs_no_index(self):
        self.backend.segment_instances.return_value = self.masks[:1]
        result = api.transform_objects(self.image, 'person', high.horizontal_flip,
                                       segmenter=self.backend, geometry=True)
        np.testing.assert_array_equal(result, self.image)

    def test_no_matches_and_out_of_range_indices(self):
        self.backend.segment.return_value = np.zeros((20, 20), dtype=bool)
        self.backend.segment_instances.return_value = self.masks[:0]
        for geometry in (False, True):
            function = Mock()
            result = api.transform_objects(self.image, 'dog', function, segmenter=self.backend, geometry=geometry)
            function.assert_not_called()
            self.assertIsNot(result, self.image)
            np.testing.assert_array_equal(result, self.image)
        with self.assertRaisesRegex(ValueError, 'outside the detected'):
            api.transform_objects(self.image, 'dog', high.grayscale, segmenter=self.backend, instance=0)

    def test_invalid_options_do_not_load_models(self):
        cases = [{'word': ''}, {'instance': -1}, {'instance': True}, {'instance': 1.5},
                 {'instance': 'left'}, {'transform': 'scale'}, {'geometry': 1},
                 {'geometry': True, 'background': Image.new('RGB', (1, 1))}]
        with patch('image_transforms.objects._default_segmenter') as default:
            for options in cases:
                kwargs = dict(word='person', transform=high.grayscale)
                kwargs.update(options)
                with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                    api.transform_objects(self.image, **kwargs)
            default.assert_not_called()

    def test_default_backend_and_rgb_input(self):
        with patch('image_transforms.objects._default_segmenter', return_value=self.backend) as default:
            api.transform_objects(self.image.convert('RGBA'), 'person', high.noise, seed=42)
            default.assert_called_once_with()
        self.assertEqual(self.backend.segment.call_args.args[0].mode, 'RGB')

    def test_invalid_instance_backends(self):
        for masks in [self.masks.astype(np.uint8), self.masks[:, :10], self.masks[0], []]:
            self.backend.segment_instances.return_value = masks
            with self.subTest(shape=np.shape(masks)), self.assertRaises(ValueError):
                api.transform_objects(self.image, 'person', high.grayscale, segmenter=self.backend, instance=0)
        with self.assertRaisesRegex(TypeError, 'segment_instances'):
            api.transform_objects(self.image, 'person', api.scale, segmenter=object(), geometry=True)
        self.backend.segment_instances.return_value = self.masks
        with self.assertRaises(ValueError):
            api.transform_objects(self.image, 'person', high.grayscale, segmenter=self.backend, instance=2)

    def test_grounded_sam_union_is_derived_from_instance_masks(self):
        backend = api.GroundedSAM.__new__(api.GroundedSAM)
        backend.segment_instances = Mock(return_value=self.masks)
        np.testing.assert_array_equal(backend.segment(self.image, 'person'), self.masks.any(axis=0))
        backend.segment_instances.assert_called_once_with(self.image, 'person')
        backend.segment_instances.return_value = self.masks[:0]
        result = backend.segment(self.image, 'dog')
        self.assertEqual(result.shape, (20, 20))
        self.assertEqual(result.dtype, np.bool_)
        self.assertFalse(result.any())


if __name__ == '__main__':
    unittest.main()
