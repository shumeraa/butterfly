"""Run the real generate.py CLI with small, model-free cached fixtures."""
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_cached_workflow import prepare


ROOT = Path(__file__).resolve().parents[1]


class CachedWorkflowTests(unittest.TestCase):
    def test_first_run_replay_and_recompute(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / 'fixture'
            output = Path(tmp) / 'output'
            source = Path(tmp) / 'test_photo.jpg'
            Image.new('RGB', (4, 4), 'red').save(source)
            prepare(source,
                    ROOT / 'examples/metamorphic/labels.json', fixture)
            command = [sys.executable, 'generate.py', '--dummy_execution',
                       '--config_path', str(fixture / 'config.yaml'),
                       '--config_options', f'output_dir={output}']
            def run(*flags):
                result = subprocess.run(command + list(flags), cwd=ROOT,
                                        text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                with (output / 'data.p').open('rb') as stream:
                    return result.stdout, pickle.load(stream)
            _, first = run('--code_path', str(fixture / 'cached_programs.p'))
            audit_path = output / 'dummy_execution.jsonl'
            audit = audit_path.read_text()
            source = json.loads((fixture / 'labels.json').read_text())
            manifest = json.loads((fixture / 'custom_tests.json').read_text())
            counts = [json.loads(line)['count'] for line in audit.splitlines()]
            self.assertEqual(counts[0], len(first) * 4)
            self.assertEqual(counts[1], sum(map(len, manifest.values())) * 3)
            self.assertEqual(counts[2], sum(len(record['max_accuracy_code']) for record in first))
            self.assertEqual(len(first), len(source))
            for record in first:
                results = record['unit_test_results'][0]
                self.assertEqual(results[3]['acc'], 0.)  # Broken baseline fails all tests.
                self.assertTrue(all(test['baseline_error'] for test in results[3]['results']))
                self.assertGreaterEqual(len(record['selected_unit_tests']), 1)
                self.assertLessEqual(len(record['selected_unit_tests']), 3)
                self.assertIn('original_program_outputs', record)
                self.assertIn('candidate_unit_tests', record)
            stdout, replay = run('--reload', '1', '--execute_unit_tests', '0', '--recompute_unit_tests', '0')
            self.assertIn('Unit tests already executed', stdout)
            self.assertEqual(audit_path.read_text(), audit)
            stdout, recomputed = run('--reload', '1', '--execute_unit_tests', '0', '--recompute_unit_tests', '1')
            self.assertIn('Recomputing unit tests', stdout)
            self.assertEqual(audit_path.read_text(), audit)
            for original, cached, rescored in zip(first, replay, recomputed):
                self.assertEqual(original['unit_test_results'], cached['unit_test_results'])
                self.assertEqual(original['unit_test_results'], rescored['unit_test_results'])

    def test_dummy_backend_rejects_non_fixture_programs(self):
        from scripts.dummy_runtime import _literal_output
        with self.assertRaises(ValueError):
            _literal_output("import os\nos.system('echo unexpected')")
        self.assertEqual(_literal_output("    return 'left'"), 'left')


if __name__ == '__main__':
    unittest.main()
