"""CPU checks for candidate baselines and the actual evaluation/training blocks."""
import ast
import copy
import io
import pickle
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from unit_test_generation.metamorphic.baselines import (
    collect_baselines, instrument_tests, resolve_answer, failed_baseline_test, answer_options, score_expected,
    SKIP, baseline_error_result, is_skipped, mean_scored,
)


class Dataset:
    def __init__(self, image):
        self.data = [dict(index=0, sample_id='q1', question='Which side?',
                          answer='dataset truth', image_path=str(image),
                          generated_code={0: ['candidate-a', 'candidate-b', 'broken']})]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return copy.deepcopy(self.data[index])


class BaselineTests(unittest.TestCase):
    def test_answer_rules(self):
        self.assertEqual(resolve_answer('', 'left'), 'left')
        self.assertEqual(resolve_answer({'left': 'right'}, 'left'), 'right')
        self.assertIs(resolve_answer({'left': 'right'}, 'above'), SKIP)   # no matching key: nothing to check
        self.assertEqual(resolve_answer({}, 'left'), 'left')
        self.assertEqual(resolve_answer({'any': 'red'}, 'left'), 'red')
        self.assertEqual(resolve_answer({'left': 'right', 'any': 'red'}, 'left'), 'right')
        self.assertEqual(resolve_answer({'any': ['gray', 'black', 'white']}, 'red'),
                         ['gray', 'black', 'white'])
        accuracy = lambda p, a: float(p == a)
        self.assertEqual(score_expected('black', ['gray', 'black', 'white'], accuracy), 1.)
        self.assertEqual(score_expected('red', ['gray', 'black', 'white'], accuracy), 0.)
        self.assertEqual(resolve_answer('explicit', 'left'), 'explicit')
        with self.assertRaises(ValueError):
            resolve_answer({'left': 123}, 'left')

    def test_unchecked_and_unmatched_rules_skip(self):
        self.assertIs(resolve_answer({'yes': 'no', 'no': 'unchecked'}, 'no'), SKIP)
        self.assertEqual(resolve_answer({'yes': 'no', 'no': 'unchecked'}, 'yes'), 'no')
        self.assertIs(resolve_answer({'any': ['unchecked']}, 'x'), SKIP)
        self.assertIs(resolve_answer('unchecked', 'x'), SKIP)
        # yes/no baseline against a left/right rule says nothing about the flipped answer
        self.assertIs(resolve_answer({'left': 'right', 'right': 'left'}, 'no'), SKIP)
        self.assertIs(resolve_answer({'left': 'right', 'right': 'left'}, 'yes'), SKIP)

    def test_baseline_answer_is_normalized_before_lookup(self):
        rule = {'left': 'right', 'right': 'left'}
        self.assertEqual(resolve_answer(rule, 'Left '), 'right')
        self.assertEqual(resolve_answer(rule, 'left side'), 'right')
        self.assertEqual(resolve_answer(rule, 'on the right'), 'left')
        self.assertIs(resolve_answer(rule, 'left or right'), SKIP)   # ambiguous
        self.assertIs(resolve_answer(rule, 'yes, the man is on the left.'), SKIP)   # too long to trust

    def test_skipped_tests_are_excluded_from_the_mean(self):
        record = dict(index=0, sample_id='q1', unit_test_metadata=[{'source': 'metamorphic'}] * 2)
        tests = [('hflip', {'yes': 'no', 'no': 'unchecked'}), ('vflip', '')]
        baseline = {('q1', 0, 0): dict(output='no', error=None), ('q1', 0, 1): dict(output='yes', error=None)}
        _, skipped = instrument_tests(record, tests, 0, 0, True, baseline)
        self.assertEqual(list(skipped), [0]); self.assertTrue(is_skipped(skipped[0]))
        _, ran = instrument_tests(record, tests, 0, 1, True, baseline)
        self.assertEqual(ran, {})
        results = [baseline_error_result(skipped[0]), dict(acc=1.0, results=[])]
        self.assertEqual(mean_scored(results), 1.0)
        self.assertEqual(mean_scored([baseline_error_result(skipped[0])]), 0.0)   # nothing scored
        failed = baseline_error_result('Baseline execution failed: x')
        self.assertEqual(mean_scored([failed, dict(acc=1.0, results=[])]), 0.5)   # a failed baseline still counts

    def test_cache_and_candidate_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'original.png'
            image.write_bytes(b'source')
            dataset = Dataset(image)
            config = dict(seed=42, execution={})
            calls = []
            def execute(tasks, config, use_fixed_code=None):
                calls.append(tasks)
                return {task['sample_id']: dict(output='left', error=None) for task in tasks}
            shared = collect_baselines(dataset, 0, config, execute, lambda code: code)
            self.assertEqual(len(calls[0]), 3)
            self.assertIn(('q1', 0, 1), shared)
            collect_baselines(dataset, 0, config, execute, lambda code: code)
            self.assertEqual(len(calls), 1)
            dataset.data[0]['unit_test_results'] = {0: 'stale'}
            dataset.data[0]['generated_code'][0][1] = 'new-candidate'
            collect_baselines(dataset, 0, config, execute, lambda code: code)
            self.assertEqual(len(calls[-1]), 1)
            self.assertNotIn(0, dataset.data[0]['unit_test_results'])
            image.write_bytes(b'changed-source')
            collect_baselines(dataset, 0, config, execute, lambda code: code)
            self.assertEqual(len(calls[-1]), 3)
            dataset.data[0]['generated_code'][1] = ['candidate-a']
            collect_baselines(dataset, 1, config, execute, lambda code: code)
            self.assertEqual(len(calls[-1]), 1)

    def test_missing_baseline_fails_and_rules_stay_shared(self):
        record = dict(index=0, sample_id='q1', unit_test_metadata=[{'source': 'metamorphic'}])
        tests = [('flip', {'left': 'right'})]
        resolved, errors = instrument_tests(record, tests, 0, 0, True, {})
        self.assertEqual(tests, [('flip', {'left': 'right'})])
        self.assertIn('Missing original execution', errors[0])
        self.assertEqual(failed_baseline_test(errors[0])['acc'], 0)
        with self.assertRaisesRegex(ValueError, 'metamorphic_baselines'):
            instrument_tests(record, tests, 0, 0, False, {})

    def test_executor_exception_and_missing_result_are_baseline_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Dataset(Path(tmp) / 'image.png')
            config = dict(seed=42, execution={})
            def broken_executor(*args, **kwargs):
                raise RuntimeError('execution service failed')
            shared = collect_baselines(dataset, 0, config, broken_executor, lambda code: code)
            self.assertIn('execution service failed', shared['q1', 0, 0]['error'])
            dataset.data[0].pop('original_program_outputs')
            shared = collect_baselines(dataset, 0, config, lambda *a, **k: {}, lambda code: code)
            self.assertIn('no result', shared['q1', 0, 0]['error'])

    def test_loader_and_append_keep_mapping_rules(self):
        import json
        from PIL import Image
        from apply_transformations.apply import append_custom_unit_tests
        from unit_test_generation.suites import build_suite
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'image.png'
            Image.new('RGB', (2, 2)).save(image)
            dataset = Dataset(image)
            mapping = {'left': 'right', 'right': 'left'}
            cases = {'q1': [{'description': 'flip', 'answer': mapping,
                             'image_paths': [str(image)]}]}
            tests_path = Path(tmp) / 'tests.json'
            tests_path.write_text(json.dumps(cases))
            config = dict(seed=42, output_dir=tmp, execution={'metamorphic_baselines': True},
                          unit_test_generation={'use_program': False},
                          unit_test_sampling={'num_unit_tests': 5},
                          unit_test_suite={'mode': 'metamorphic', 'metamorphic': {'tests_path': str(tests_path)}})
            build_suite(dataset, config, lambda: self.fail('Unexpected ViUniT call'))
            self.assertEqual(dataset.data[0]['selected_unit_tests'], [('flip', mapping)])
            self.assertEqual(append_custom_unit_tests(dataset.data, cases), 0)
            candidate_cases = copy.deepcopy(cases)
            candidate_cases['q1'][0]['description'] = 'other flip'
            self.assertEqual(append_custom_unit_tests(dataset.data, candidate_cases), 1)
            self.assertEqual(dataset.data[0]['unit_test_metadata'][-1]['source'], 'metamorphic')
            self.assertEqual(dataset.data[0]['selected_unit_tests'][-1][1], mapping)

    def test_execution_and_recomputation_in_both_entrypoints(self):
        # Execute the real unit-test blocks without importing model dependencies
        # or running their command-line setup. Providers and executors are fakes.
        for filename in ('generate.py', 'train.py'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                image = Path(tmp) / 'original.png'
                image.write_bytes(b'source')
                dataset = Dataset(image)
                config = dict(do_unit_test=True, seed=42,
                              unit_test_generation=dict(use_program=False),
                              execution=dict(synonym_checker=False, image_agg='mean',
                                             error_penalty=dict(syntax=.1, runtime=.1)))
                def baseline_execute(tasks, config, use_fixed_code=None):
                    answers = ['left', 'right', None]
                    return {task['sample_id']: dict(output=answer,
                            error='SyntaxError: broken program' if answer is None else None)
                            for task, answer in zip(tasks, answers)}
                shared = collect_baselines(dataset, 0, config, baseline_execute, lambda code: code)
                def build_suite(dataset, config, provider):
                    dataset.data[0].update(
                        selected_unit_tests=[('native', 'red'), ('flip', {'left': 'right', 'right': 'left'}),
                                             ('invariant', ''), ('alternatives', {'any': ['left', 'right']})],
                        unit_test_images=[[str(image)]] * 4,
                        unit_test_metadata=[{'source': 'viunit'}] + [{'source': 'metamorphic'}] * 3)
                calls = []
                def execute(tasks, config, use_fixed_code=None):
                    calls.append(tasks)
                    return {task['sample_id']: dict(output='red' if '_0_' in task['sample_id'][2:] else 'right',
                                                   error=None) for task in tasks}
                source = Path(filename).read_text()
                tree = ast.parse(source)
                loop = next(node for node in tree.body if isinstance(node, ast.For)
                            and isinstance(node.target, ast.Name) and node.target.id == 'iter')
                block = next(node for node in loop.body if isinstance(node, ast.If)
                             and ast.unparse(node.test) == "this_config['do_unit_test']")
                program = compile(ast.Module(body=[block], type_ignores=[]), filename, 'exec')
                namespace = dict(dataset=dataset, this_config=config, iter=0, baseline_enabled=True,
                                 original_answers=shared, build_suite=build_suite, instrument_tests=instrument_tests,
                                 answer_options=answer_options, score_expected=score_expected,
                                 failed_baseline_test=failed_baseline_test, baseline_error_result=baseline_error_result,
                                 mean_scored=mean_scored, copy=copy,
                                 pickle=SimpleNamespace(dump=lambda data, stream: stream.close()),
                                 open=open, save_file=str(Path(tmp) / 'data.p'),
                                 log_file=io.StringIO(), tqdm=lambda items: items,
                                 Counter=Counter, extract_python_code=lambda code: code,
                                 get_penalty=lambda *args: 0, acc_fn=lambda p, g: float(p == g),
                                 code_execution=execute,
                                 args=SimpleNamespace(execute_unit_tests=True, recompute_unit_tests=True))
                exec(program, namespace)
                record = dataset.data[0]
                self.assertEqual(record['candidate_unit_tests'][0][0][1], ('flip', 'right'))
                self.assertEqual(record['candidate_unit_tests'][0][1][1], ('flip', 'left'))
                results = record['unit_test_results'][0]
                self.assertEqual(results[0]['results'][1]['acc'], 1)
                self.assertEqual(results[1]['results'][1]['acc'], 0)
                self.assertEqual(results[0]['results'][3]['acc'], 1)
                self.assertEqual(results[1]['results'][3]['acc'], 1)
                alternatives = [task for task in calls[0] if task['sample_id'] == '0_0_3_0']
                self.assertEqual(alternatives[0]['possible_answers'], ['left', 'right'])
                self.assertEqual(results[2]['results'][1]['acc'], 0)
                self.assertIn('SyntaxError', results[2]['results'][1]['error'])
                self.assertFalse(any(task['sample_id'].startswith('0_2_1_') for task in calls[0]))
                self.assertEqual(record['selected_unit_tests'][1][1]['left'], 'right')
                namespace['args'].execute_unit_tests = False
                exec(program, namespace)
                self.assertEqual(len(calls), 1)  # Recompute scores from stored outputs.
                self.assertEqual(record['unit_test_results'][0][2]['results'][1]['acc'], 0)


if __name__ == '__main__':
    unittest.main()
