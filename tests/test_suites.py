"""CPU-only dispatcher checks; no generation services or models required."""
import copy
import json

from PIL import Image
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from unit_test_generation.metamorphic import MetamorphicTest
from unit_test_generation.suites import build_suite, invalidate_suite_cache, suite_settings


class Dataset:
    def __init__(self, image):
        self.data = [dict(index=0, sample_id='q1', text='What color?',
                          question='What color?', answer='red', image_path=str(image))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return copy.deepcopy(self.data[index])


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image = Path(self.temp.name) / 'image.png'
        Image.new('RGB', (8, 6), 'red').save(self.image)
        self.dataset = Dataset(self.image)
        self.config = dict(seed=42, output_dir=self.temp.name,
                           unit_test_generation=dict(use_program=False),
                           unit_test_sampling=dict(num_unit_tests=5),
                           unit_test_suite=dict(mode='viunit', viunit_count=None,
                                                metamorphic_count=2, metamorphic={}))
        self.calls = 0

    def viunit(self):
        self.calls += 1
        d = self.dataset.data[0]
        count = self.config['unit_test_sampling']['num_unit_tests']
        d['unit_tests'] = [('caption', 'red')] * count
        d['selected_unit_tests'] = list(d['unit_tests'])
        d['unit_test_images'] = [[str(self.image)] for _ in range(count)]

    def build(self):
        build_suite(self.dataset, self.config, self.viunit)

    def test_default_viunit_cache_and_invalidation(self):
        self.build()
        d = self.dataset.data[0]
        d['unit_test_results'] = {0: 'cached'}
        self.build()
        self.assertEqual(self.calls, 1)
        self.assertIn('unit_test_results', d)
        self.config['seed'] += 1
        self.build()
        self.assertEqual(self.calls, 2)
        self.assertNotIn('unit_test_results', d)
        d.pop('unit_test_images')  # --load_images=0 requires rebuilding
        self.build()
        self.assertEqual(self.calls, 3)

    def test_hybrid_alignment_and_idempotence(self):
        self.config['unit_test_suite'].update(mode='hybrid', viunit_count=3)
        custom = [MetamorphicTest(self.image, 'blue', 'changed color', {'operation': 'hsv'})]
        with patch('unit_test_generation.suites.metamorphic.generate_tests', return_value=custom) as generate:
            self.build()
            self.build()
            generate.assert_called_once()
            self.assertNotIn('program', generate.call_args.kwargs)
        d = self.dataset.data[0]
        self.assertEqual(len(d['selected_unit_tests']), 4)
        self.assertEqual(len(d['unit_test_images']), 4)
        self.assertEqual(d['selected_unit_tests'][-1], ('changed color', 'blue'))
        self.assertEqual(d['unit_test_metadata'][-1]['source'], 'metamorphic')
        self.assertEqual(self.config['unit_test_sampling']['num_unit_tests'], 5)

    def test_metamorphic_skips_viunit_and_invalidates_on_mode_change(self):
        self.build()
        self.config['unit_test_suite']['mode'] = 'metamorphic'
        with patch('unit_test_generation.suites.metamorphic.generate_tests',
                   return_value=[MetamorphicTest(self.image, 'red', 'flip')]):
            self.build()
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.dataset.data[0]['selected_unit_tests'], [('flip', 'red')])

    def test_real_metamorphic_generation_and_cache(self):
        self.config['unit_test_suite'].update(mode='metamorphic', metamorphic_count=2,
            metamorphic={'transformations': ['horizontal_flip', 'noise']})
        self.build()
        d = self.dataset.data[0]
        self.assertEqual(self.calls, 0)
        self.assertEqual(len(d['selected_unit_tests']), 2)
        self.assertTrue(all(answer == '' for _, answer in d['selected_unit_tests']))
        self.assertTrue(all(m['source'] == 'metamorphic' for m in d['unit_test_metadata']))
        paths = [Path(group[0]) for group in d['unit_test_images']]
        contents = [path.read_bytes() for path in paths]
        self.build()
        self.assertEqual(contents, [path.read_bytes() for path in paths])
        self.assertTrue(all(path.is_absolute() and path.is_file() for path in paths))
        with Image.open(self.image) as source:
            self.assertEqual(source.size, (8, 6))

    def test_predictions_hybrid_and_file_change_invalidates_feedback(self):
        predictions = Path(self.temp.name) / 'predictions.json'
        def write(safe):
            predictions.write_text(json.dumps([{'question_id': 'q1', 'parsed': {
                'horizontal_flip': {'relation': 'invariant' if safe else 'unsafe'},
                'grayscale': {'relation': 'unsafe' if safe else 'variant'},
            }}]))
        write(True)
        self.config['unit_test_suite'].update(mode='hybrid', viunit_count=3,
            metamorphic={'predictions_path': str(predictions)})
        self.build()
        d = self.dataset.data[0]
        self.assertEqual(d['selected_unit_tests'][-1], ('horizontal_flip', ''))
        self.assertEqual(len(d['unit_test_metadata']), 4)
        d['unit_test_results'] = {0: 'old'}
        write(False)
        invalidate_suite_cache(self.dataset, self.config)
        self.assertNotIn('unit_test_results', d)
        self.build()
        self.assertEqual(d['selected_unit_tests'][-1], ('grayscale', ''))
        self.assertEqual(self.calls, 2)

    def test_prebuilt_tests_with_blank_answers(self):
        tests = Path(self.temp.name) / 'tests.json'
        tests.write_text(json.dumps({'q1': [{'description': 'flipped', 'answer': '',
                                          'image_paths': [str(self.image)]}]}))
        self.config['unit_test_suite'].update(mode='metamorphic',
            metamorphic={'tests_path': str(tests)})
        self.build()
        self.assertEqual(self.dataset.data[0]['selected_unit_tests'], [('flipped', '')])
        self.assertEqual(self.calls, 0)

    def test_export_round_trip_and_blank_answer_append(self):
        from apply_transformations.apply import apply_transformations, append_custom_unit_tests
        exported = Path(self.temp.name) / 'exported.json'
        predictions = json.dumps([{'question_id': 'q1', 'parsed': {
            'horizontal_flip': {'relation': 'invariant'},
            'grayscale': {'relation': 'unsafe'},
        }}])
        original = self.image.read_bytes()
        cases = apply_transformations(predictions, str(self.image),
                                      str(Path(self.temp.name) / 'variants'), str(exported))
        self.assertEqual(cases['q1'][0]['answer'], '')
        self.assertEqual(cases['q1'][0]['description'], 'horizontal_flip')
        self.assertEqual(self.image.read_bytes(), original)
        self.build()
        self.assertEqual(append_custom_unit_tests(self.dataset.data, cases), 1)
        self.assertEqual(append_custom_unit_tests(self.dataset.data, cases), 0)
        self.config['unit_test_suite'].update(mode='metamorphic',
            metamorphic={'tests_path': str(exported)})
        self.build()
        self.assertEqual(self.dataset.data[0]['selected_unit_tests'], [('horizontal_flip', '')])

    def test_unsafe_predictions_fail_before_diffusion(self):
        self.config['unit_test_suite'].update(mode='hybrid',
            metamorphic={'transformations': []})
        with self.assertRaises(ValueError):
            self.build()
        self.assertEqual(self.calls, 0)
        self.assertNotIn('unit_test_suite_key', self.dataset.data[0])

    def test_invalidation_before_feedback(self):
        self.build()
        self.dataset.data[0]['unit_test_results'] = {0: 'old feedback'}
        self.config['unit_test_suite']['mode'] = 'metamorphic'
        invalidate_suite_cache(self.dataset, self.config)
        self.assertNotIn('unit_test_results', self.dataset.data[0])
        self.assertNotIn('unit_test_images', self.dataset.data[0])

    def test_invalid_settings(self):
        self.config['unit_test_suite']['mode'] = 'unknown'
        with self.assertRaises(ValueError):
            suite_settings(self.config)
        self.config['unit_test_suite']['mode'] = 'hybrid'
        self.config['unit_test_generation']['use_program'] = True
        with self.assertRaises(ValueError):
            suite_settings(self.config)


if __name__ == '__main__':
    unittest.main()
