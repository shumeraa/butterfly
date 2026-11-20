"""Pixel-level tests for custom image creation and its suite integration."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from apply_transformations.apply import create_test_manifest
from apply_transformations.materialize import materialize_selected_tests
from image_transforms import fill_objects, occlude_objects
from unit_test_generation.suites import build_suite, invalidate_suite_cache


class Segmenter:
    def segment(self, image, word):
        mask = np.zeros((image.height, image.width), dtype=bool)
        mask[1, 1] = mask[2, 2] = True
        return mask

    def detect_boxes(self, image, word):
        return np.array([[1, 1, 3, 3]], dtype=float)


class Dataset:
    def __init__(self, path):
        self.data = [dict(index=0, sample_id='q1', question='Which side?', text='Which side?',
                          answer='left', image_path=str(path))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return copy.deepcopy(self.data[index])


class MaterializedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'original.png'
        self.pixels = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        Image.fromarray(self.pixels).save(self.source)

    def test_independent_mr1_images_and_saved_rules(self):
        tests = [dict(description=name, transformation=name, parameters={}, answer=rule)
                 for name, rule in [('hflip', {'left': 'right'}), ('vflip', ''), ('grayscale', '')]]
        before = self.source.read_bytes()
        original_tests = copy.deepcopy(tests)
        with patch('image_transforms.masking._default_segmenter') as backend:
            result = materialize_selected_tests(self.source, tests, self.root / 'saved')
            backend.assert_not_called()
        paths = [Path(test['image_paths'][0]) for test in result]
        self.assertEqual(len(set(paths)), 3)
        self.assertTrue(all(path.is_absolute() and path.is_file() for path in paths))
        with Image.open(paths[0]) as image:
            np.testing.assert_array_equal(image, self.pixels[:, ::-1])
        with Image.open(paths[1]) as image:
            np.testing.assert_array_equal(image, self.pixels[::-1])
        with Image.open(paths[2]) as image:
            array = np.asarray(image)
            np.testing.assert_array_equal(array[:, :, 0], array[:, :, 1])
            np.testing.assert_array_equal(array[:, :, 1], array[:, :, 2])
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(tests, original_tests)
        self.assertEqual(result[0]['answer'], {'left': 'right'})
        self.assertEqual(json.loads((paths[0].parent / 'tests.json').read_text()), result)

    def test_color_fill_and_box_occlusion_preserve_alpha(self):
        source = Image.fromarray(self.pixels).convert('RGBA')
        source.putalpha(Image.fromarray(np.arange(20, dtype=np.uint8).reshape(4, 5)))
        original = np.asarray(source).copy()
        backend = Segmenter()
        mask = backend.segment(source, 'cup')
        result = np.asarray(fill_objects(source, 'cup', 'red', segmenter=backend))
        np.testing.assert_array_equal(result[mask, :3], np.tile([255, 0, 0], (2, 1)))
        np.testing.assert_array_equal(result[~mask], original[~mask])
        np.testing.assert_array_equal(result[:, :, 3], original[:, :, 3])
        occluded = np.asarray(occlude_objects(source, 'cup', segmenter=backend))
        self.assertTrue(np.all(occluded[1:3, 1:3, :3] == 255))
        # The box includes background pixels outside the segmentation mask.
        self.assertTrue(np.all(occluded[1, 2, :3] == 255))
        np.testing.assert_array_equal(occluded[:, :, 3], original[:, :, 3])
        np.testing.assert_array_equal(source, original)

    def test_empty_targets_fail_instead_of_creating_false_tests(self):
        class Empty:
            def segment(self, image, word):
                return np.zeros((image.height, image.width), dtype=bool)

            def detect_boxes(self, image, word):
                return np.empty((0, 4))
        for operation, parameters in [('fill_color', {'item': 'cup', 'color': 'red'}),
                                      ('occlude_bbox', {'item': 'cup'})]:
            folder = self.root / operation
            with self.assertRaisesRegex(ValueError, 'No object matched'):
                materialize_selected_tests(self.source, [dict(description=operation,
                    transformation=operation, parameters=parameters, answer='')], folder,
                    segmenter=Empty())
            self.assertEqual(list(folder.glob('*.png')), [])
            self.assertFalse((folder / 'tests.json').exists())

    def test_labels_pipeline_selects_applies_and_rebuilds_missing_images(self):
        labels = [['q1', dict(hflip={'equivariant': {'left': 'right'}},
                             vflip='invariant', grayscale='invariant', mr2={
            'cup': dict(fill_color={'red': {'equivariant': {'any': 'red'}}},
                        occlude_bbox={'equivariant': {'any': 'no'}})})]]
        labels_path = self.root / 'labels.json'
        labels_path.write_text(json.dumps(labels))
        config = dict(seed=42, output_dir=str(self.root / 'run'),
                      unit_test_generation={'use_program': False},
                      unit_test_sampling={'num_unit_tests': 5},
                      unit_test_suite={'mode': 'metamorphic', 'metamorphic_count': 5,
                                       'metamorphic': {'labels_path': str(labels_path)}})
        dataset = Dataset(self.source)
        before = self.source.read_bytes()
        with patch('image_transforms.masking._default_segmenter', return_value=Segmenter()):
            build_suite(dataset, config, lambda: self.fail('Unexpected ViUniT provider'))
            record = dataset.data[0]
            self.assertEqual([test[0] for test in record['selected_unit_tests']],
                             ['hflip', 'cup turned red', 'cup, occluded bounding box'])
            paths = [Path(group[0]) for group in record['unit_test_images']]
            self.assertTrue(all(str(path).startswith(str((self.root / 'run').resolve())) for path in paths))
            self.assertEqual(len(record['unit_test_metadata']), 3)
            with Image.open(paths[1]) as image:
                self.assertEqual(image.getpixel((1, 1)), (255, 0, 0))
            with patch('apply_transformations.materialize.materialize_selected_tests') as apply:
                build_suite(dataset, config, lambda: None)
                apply.assert_not_called()
            record['unit_test_results'] = {0: 'old feedback'}
            paths[1].unlink()
            invalidate_suite_cache(dataset, config)
            self.assertNotIn('unit_test_results', record)
            build_suite(dataset, config, lambda: None)
            self.assertTrue(paths[1].is_file())
        self.assertEqual(self.source.read_bytes(), before)

    def test_selected_test_records_materialize_instead_of_using_placeholder_paths(self):
        cases = {'q1': [dict(description='flip', transformation='hflip', parameters={},
                             answer={'left': 'right'}, image_paths=['does-not-exist.png'])]}
        tests_path = self.root / 'tests.json'
        tests_path.write_text(json.dumps(cases))
        from unit_test_generation.metamorphic.generator import generate_tests
        tests = generate_tests(question_id='q1', question='Where?', original_image_path=self.source,
                               original_answer='left', num_tests=3, output_dir=self.root / 'saved', seed=42,
                               config={'tests_path': str(tests_path)})
        with Image.open(tests[0].image_path) as image:
            np.testing.assert_array_equal(image, self.pixels[:, ::-1])
        self.assertEqual(tests[0].expected_answer, {'left': 'right'})

    def test_mr1_only_filters_mr2_before_selection(self):
        from unit_test_generation.metamorphic.generator import generate_tests
        labels = [['q1', dict(hflip='invariant', vflip='invariant', grayscale='invariant',
            mr2={'cup': dict(fill_color={'red': {'equivariant': {'any': 'red'}}},
                            occlude_bbox={'equivariant': {'any': 'no'}})})]]
        labels_path = self.root / 'labels.json'
        labels_path.write_text(json.dumps(labels))
        with patch('image_transforms.masking._default_segmenter') as backend:
            tests = generate_tests(question_id='q1', question='Where?', original_image_path=self.source,
                original_answer='left', num_tests=3, output_dir=self.root / 'mr1', seed=42,
                config={'labels_path': str(labels_path), 'mr2_enabled': False})
            backend.assert_not_called()
        self.assertEqual({test.metadata['transformation'] for test in tests}, {'hflip', 'vflip', 'grayscale'})
        self.assertTrue(all(test.image_path.is_file() for test in tests))

    def test_standalone_export_saves_real_images(self):
        result = create_test_manifest(json.dumps([['q1', {'hflip': 'invariant', 'mr2': None}]]),
            str(self.source), str(self.root / 'export'), str(self.root / 'manifest.json'))
        saved = Path(result['q1'][0]['image_paths'][0])
        self.assertTrue(saved.is_file())
        self.assertNotEqual(saved, self.source)
        with Image.open(saved) as image:
            np.testing.assert_array_equal(image, self.pixels[:, ::-1])


if __name__ == '__main__':
    unittest.main()
