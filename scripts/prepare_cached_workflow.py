"""Prepare a photo-backed dummy run from LLM labels, not pre-generated tests."""
import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apply_transformations.apply import create_test_manifest
from apply_transformations.labels import tests_from_labels


def candidate_answers(tests):
    """Derive two dummy answers from selected rules, plus an unmapped baseline."""
    keys, targets = [], []
    for test in tests:
        rule = test['answer']
        if isinstance(rule, dict):
            keys.extend(key for key in rule if key != 'any')
            for value in rule.values():
                targets.extend(value if isinstance(value, list) else [value])
    choices = list(dict.fromkeys(keys + targets + ['yes', 'no']))
    return choices[:2] + ['__unmapped__']


def prepare(image, labels_path, destination, *, max_tests=3, seed=42):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    photo = destination / 'dummy.jpg'
    if Path(image).resolve() != photo.resolve():
        shutil.copy2(image, photo)
    source = Path(labels_path).read_text(encoding='utf-8')
    labels = json.loads(source)
    all_tests = tests_from_labels(labels)
    # Keep the original labels unchanged so every selected rule is traceable.
    (destination / 'labels.json').write_text(source, encoding='utf-8')
    tests = create_test_manifest(source, str(photo), str(destination / 'transformed'),
        str(destination / 'custom_tests.json'), num_tests=max_tests, seed=seed, dummy_images=True)
    annotations, programs, excluded = [], [], []
    for question_id, cases in list(tests.items()):
        if not cases:
            excluded.append(question_id)
            del tests[question_id]
            continue
        outputs = candidate_answers(cases)
        # Labels contain neither original questions/images nor ground-truth answers.
        annotations.append(dict(questionId=question_id, imageId='dummy',
            question=f'Dummy question for {question_id}', answer=outputs[0]))
        codes = [f'def execute_command(image):\n    return {answer!r}\n' for answer in outputs]
        codes.append('def execute_command(image):\n    raise RuntimeError("intentional baseline failure")\n')
        programs.append(dict(generated_code={0: codes}))
    if not annotations:
        raise ValueError('No questions have invariant or equivariant transformations')
    (destination / 'annotations.json').write_text(json.dumps(annotations, indent=2) + '\n')
    (destination / 'custom_tests.json').write_text(json.dumps(tests, indent=2) + '\n')
    with (destination / 'cached_programs.p').open('wb') as stream:
        pickle.dump(programs, stream)
    report = dict(source='labels.json', seed=seed, max_tests_per_question=max_tests,
                  source_questions=len(labels), included_questions=len(annotations),
                  excluded_without_valid_transformations=excluded,
                  eligible_tests=sum(map(len, all_tests.values())),
                  selected_tests=sum(map(len, tests.values())),
                  equivariant_tests=sum(case['relation'] == 'equivariant'
                                        for cases in tests.values() for case in cases),
                  candidates=len(annotations) * 4,
                  image_mode='shared dummy photo; no transformations applied',
                  ground_truth='synthetic first candidate answer; not supplied by labels')
    (destination / 'preparation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    # All smoke-run inputs are reproducible outputs, not checked-in fixtures.
    config = dict(output_dir=str(destination.parent / 'run'), do_unit_test=True,
        do_train=False, run_fixed_code=False,
        data=dict(dataset_name='GQA', dataset_args=dict(
            annotation_file=str(destination / 'annotations.json'), image_root=str(destination),
            num_samples=-1, sample_per_type=-1, sample_ids_file=None)),
        visual_program_generator=dict(model_name='dummy'),
        unit_test_generation=dict(use_program=False),
        unit_test_suite=dict(mode='metamorphic', metamorphic_count=max_tests,
            metamorphic=dict(tests_path=str(destination / 'custom_tests.json'), materialize_images=False)),
        execution=dict(metamorphic_baselines=True, feedback_max_iters=1, synonym_checker=False,
                       image_agg='mean', code_agg='vote', use_fixed_code_when_low_confidence=False))
    OmegaConf.save(OmegaConf.create(config), destination / 'config.yaml')
    print(f'Prepared {len(annotations)} questions, {len(annotations) * 4} candidates, '
          f'{report["selected_tests"]} custom tests from labels using {photo}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--labels', type=Path, default=Path('examples/metamorphic/labels.json'))
    parser.add_argument('--destination', type=Path, default=Path('output/metamorphic_smoke/inputs'))
    parser.add_argument('--max-tests', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.image, args.labels, args.destination, max_tests=args.max_tests, seed=args.seed)
