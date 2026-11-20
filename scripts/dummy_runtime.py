"""Offline adapters for generate.py's explicitly selected dummy fixture mode.

The real dispatcher, baseline cache, instrumentation and scoring loops still run.
This backend only interprets literal return/RuntimeError fixture programs, never
loads models, and does not implement arbitrary visual-program execution.
"""
import ast
import copy
import json
import random
import textwrap
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image


def load_config(base_path, config_path, options):
    config = OmegaConf.merge(OmegaConf.load(base_path), OmegaConf.load(config_path))
    if options:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(options))
    return config


def set_seed(seed):
    random.seed(seed)


def extract_python_code(code):
    tree = ast.parse(textwrap.dedent(code))
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
        return ast.unparse(tree.body[0])
    return 'def execute_command(image):\n' + textwrap.indent(textwrap.dedent(code), '    ')


def _literal_output(code):
    tree = ast.parse(textwrap.dedent(code))
    body = tree.body
    if len(body) == 1 and isinstance(body[0], ast.FunctionDef):
        body = body[0].body
    if len(body) != 1:
        raise ValueError('Dummy programs must contain exactly one literal return or RuntimeError')
    statement = body[0]
    if isinstance(statement, ast.Return) and isinstance(statement.value, ast.Constant):
        if isinstance(statement.value.value, str):
            return statement.value.value
    if isinstance(statement, ast.Raise) and isinstance(statement.exc, ast.Call):
        call = statement.exc
        if (isinstance(call.func, ast.Name) and call.func.id == 'RuntimeError'
                and len(call.args) == 1 and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str) and not call.keywords):
            raise RuntimeError(call.args[0].value)
    raise ValueError('Dummy backend accepts only string literals and RuntimeError fixtures')


def code_execution(tasks, config, use_fixed_code=None):
    results = {}
    for task in tasks:
        try:
            # Validate the shared photo path just as a real image-backed run would.
            with Image.open(task['image']) as image:
                image.verify()
            result = dict(output=_literal_output(task['codes']), error=None)
        except Exception as error:
            result = dict(output=None, error=f'{type(error).__name__}: {error}')
        result['acc'] = (float(result['output'] == task['answer'])
                         if result['error'] is None else 0.)
        results[task['sample_id']] = result
    audit = Path(config['output_dir']) / 'dummy_execution.jsonl'
    with audit.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(sample_ids=[task['sample_id'] for task in tasks],
                                     count=len(tasks))) + '\n')
    return results


class DummyDataset:
    def __init__(self, annotation_file, image_root, **kwargs):
        self.data = json.loads(Path(annotation_file).read_text())
        self.image_root = image_root
        for record in self.data:
            record['sample_id'] = record['questionId']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = copy.deepcopy(self.data[index])
        record['image_path'] = str(Path(self.image_root) / (record['imageId'] + '.jpg'))
        record['text'] = record['question']
        return record


def get_dataset_class(name):
    if name != 'GQA':
        raise ValueError('Dummy mode supports the GQA-shaped fixture only')
    return DummyDataset


def get_accuracy_fn(name):
    # Exact strings suffice for the fixture; production postprocessing is unchanged.
    return lambda predictions, answers: sum(p == a for p, a in zip(predictions, answers)) / len(answers)


def get_fixed_code(task):
    return "    return 'dummy'\n"


def get_penalty(error, syntax=.1, runtime=.1):
    return 0 if error is None else syntax if 'Compilation' in str(error) else runtime


def engine_execution(*args, **kwargs):
    raise RuntimeError('Dummy mode requires cached programs; model generation is disabled')


def write_summary(records, config):
    """Expose the relevant pickle fields as JSON for inspecting the dummy run."""
    summary = {}
    for record in records:
        summary[record['sample_id']] = dict(
            baseline_outputs=record.get('original_program_outputs', {}),
            shared_tests=record.get('selected_unit_tests', []),
            candidate_tests=record.get('candidate_unit_tests', {}),
            test_results=record.get('unit_test_results', {}),
            final_accuracy=record.get('final_accuracy'),
        )
    path = Path(config['output_dir']) / 'dummy_summary.json'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Dummy results: {path}')
