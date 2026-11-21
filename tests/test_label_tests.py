"""Label parsing, answer preservation, and equivariant-first suite selection."""
import ast
import copy
import unittest
from pathlib import Path

from apply_transformations.labels import tests_from_labels, select_label_tests


class LabelTestCreationTests(unittest.TestCase):
    def test_null_unsafe_and_nested_rules(self):
        mapping = {'any': ['gray', 'black', 'white']}
        labels = [['001', dict(hflip='unsafe', vflip=None,
            grayscale={'equivariant': mapping}, mr2={
                'cup': dict(fill_color={'red': 'invariant', 'blue': None, 'green': 'unsafe'},
                            occlude_bbox=None),
                'box': dict(fill_color=None, occlude_bbox={'equivariant': {'any': 'no'}}),
            })], ['002', dict(hflip='invariant', mr2=None)]]
        original = copy.deepcopy(labels)
        result = tests_from_labels(labels)
        self.assertEqual(list(result), ['001', '002'])
        self.assertEqual([test['description'] for test in result['001']],
                         ['grayscale', 'cup turned red', 'box, occluded bounding box'])
        self.assertEqual(result['001'][0]['answer'], mapping)
        self.assertEqual(result['001'][1]['answer'], '')
        self.assertEqual(labels, original)
        self.assertEqual(result['002'][0]['description'], 'hflip')

    def test_equivariance_has_priority_over_invariance(self):
        tests = [dict(description=f'i{i}', relation='invariant') for i in range(8)]
        tests += [dict(description=f'e{i}', relation='equivariant') for i in range(2)]
        selected = select_label_tests(tests, question_id='q1')
        self.assertEqual([test['description'] for test in selected[:2]], ['e0', 'e1'])
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[-1]['relation'], 'invariant')
        self.assertEqual(selected, select_label_tests(tests, question_id='q1'))

    def test_more_than_three_equivariant_tests(self):
        tests = [dict(description=str(i), relation='equivariant') for i in range(6)]
        tests.append(dict(description='invariant', relation='invariant'))
        selected = select_label_tests(tests, question_id='q1')
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(test['relation'] == 'equivariant' for test in selected))

    def test_fewer_than_three_and_no_valid_tests(self):
        tests = tests_from_labels([['q1', {'hflip': 'invariant', 'mr2': None}]])['q1']
        self.assertEqual(select_label_tests(tests, question_id='q1'), tests)
        self.assertEqual(select_label_tests([], question_id='q1'), [])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            tests_from_labels([['q1', {}], ['q1', {}]])

    def test_source_labels_selection_matches_rules(self):
        import json
        path = Path(__file__).resolve().parents[1] / 'examples/metamorphic/labels.json'
        candidates = tests_from_labels(json.loads(path.read_text()))
        selected_count = 0
        for question_id, tests in candidates.items():
            actual = select_label_tests(tests, question_id=question_id)
            selected_count += len(actual)
            self.assertLessEqual(len(actual), 3)
            eq_count = sum(test['relation'] == 'equivariant' for test in tests)
            self.assertEqual(sum(test['relation'] == 'equivariant' for test in actual), min(3, eq_count))
        self.assertEqual(len(candidates), 200)
        self.assertEqual(selected_count, 535)

    def test_correction_prompt_accepts_multiple_expected_answers(self):
        # Exercise the actual formatter without importing model dependencies.
        path = Path(__file__).resolve().parents[1] / 'utils.py'
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'get_visual_program_correction_prompt')
        namespace = dict(extract_python_code=lambda code: code)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        result = namespace[function.name]('Question', 'INSERT_UNIT_TEST_OUTPUTS_HERE', 'code',
            [('grayscale', ['gray', 'black', 'white'])], [{'output': 'black', 'error': None}],
            'dummy', tokenizer=object())
        self.assertIn("['gray', 'black', 'white']", result)


if __name__ == '__main__':
    unittest.main()
