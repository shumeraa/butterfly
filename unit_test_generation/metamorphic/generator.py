"""Per-question adapters for the merged image transformation pipeline."""
import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union


@dataclass
class MetamorphicTest:
    """One materialized test, reused across all programs for a question.

    image_path points to the transformed image, not the original image.
    expected_answer holds a string or answer mapping until candidate instrumentation.
    An empty string preserves the candidate baseline; a mapping changes it.
    metadata can hold the transformation name, parameters, and generation details.
    """

    image_path: Path
    expected_answer: Union[str, Dict[str, Union[str, List[str]]]]
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def generate_tests(
    *,
    question_id: str,
    question: str,
    original_image_path: Union[str, Path],
    original_answer: str,
    num_tests: int,
    output_dir: Union[str, Path],
    seed: int,
    config: Mapping[str, Any],
) -> List[MetamorphicTest]:
    """Materialize transformations or load custom tests for one question.

    labels_path selects raw LLM MR1/MR2 labels and materializes up to three tests.
    predictions_path also accepts the older question_id/parsed format.
    tests_path reads the custom-test JSON mapping produced by apply_transformations.
    Otherwise transformations selects prompt keys, defaulting to all six.
    Blank answers and mapping rules are retained for candidate instrumentation.
    """
    from apply_transformations.apply import eligible_transformations, materialize_tests
    from image_transforms import HIGH_LEVEL_TRANSFORMS

    from apply_transformations.labels import tests_from_labels, select_label_tests
    from apply_transformations.materialize import materialize_selected_tests

    inputs = {name: config.get(name) for name in ('labels_path', 'predictions_path', 'tests_path')}
    if sum(bool(value) for value in inputs.values()) > 1:
        raise ValueError('Choose one of labels_path, predictions_path, or tests_path')
    labels_path, predictions_path, tests_path = (inputs[name] for name in inputs)
    create_images = config.get('materialize_images', True)
    if not isinstance(create_images, bool):
        raise ValueError('materialize_images must be a boolean')
    mr2_enabled = config.get('mr2_enabled', True)
    if not isinstance(mr2_enabled, bool):
        raise ValueError('mr2_enabled must be a boolean')

    def eligible_cases(cases):
        # Exclude MR2 before prioritizing/sampling, so MR1 fills the full budget.
        return [case for case in cases if mr2_enabled or
                case.get('transformation') not in {'fill_color', 'occlude_bbox'}]

    payload = None
    if labels_path or predictions_path:
        payload = json.loads(Path(labels_path or predictions_path).read_text(encoding='utf-8'))
    if labels_path or (payload and isinstance(payload[0], list)):
        by_question = tests_from_labels(payload)
        if str(question_id) not in by_question:
            raise ValueError(f'No transformation labels for question {question_id}')
        cases = select_label_tests(eligible_cases(by_question[str(question_id)]), question_id=question_id,
                                   max_tests=min(3, num_tests), seed=seed)
        input_kind = 'labels_path' if labels_path else 'predictions_path'
    elif tests_path:
        cases = eligible_cases(json.loads(Path(tests_path).read_text(encoding='utf-8'))[str(question_id)])[:num_tests]
        input_kind = 'tests_path'
    else:
        cases = None
    if cases is not None:
        # Records with operations are materialized as they become executable tests.
        # Old path-only records continue to refer to pre-created image files.
        operations = [case for case in cases if 'transformation' in case]
        if create_images and operations:
            saved = iter(materialize_selected_tests(
                original_image_path, operations, output_dir, seed=seed,
                segmenter_config=config.get('segmenter', {}),
            ))
            cases = [next(saved) if 'transformation' in case else case for case in cases]
        tests = []
        for case in cases:
            paths = case.get('image_paths', [str(original_image_path)] if not create_images else [])
            if not isinstance(paths, list) or len(paths) != 1:
                raise ValueError('Metamorphic tests require exactly one transformed image per test')
            tests.append(MetamorphicTest(
                image_path=Path(paths[0]).resolve(), expected_answer=case.get('answer', ''),
                description=case['description'], metadata={'input': input_kind, **{key: case[key] for key in
                    ('transformation', 'relation', 'parameters', 'seed', 'original_image_path') if key in case}},
            ))
        return tests

    transformations = config.get('transformations', HIGH_LEVEL_TRANSFORMS)
    if predictions_path:
        predictions = payload
        matches = [record for record in predictions
                   if str(record['question_id']) == str(question_id)]
        if len(matches) != 1:
            raise ValueError(f'Expected one prediction for question {question_id}; found {len(matches)}')
        transformations = eligible_transformations(matches[0]['parsed'])
    if isinstance(transformations, str):
        raise ValueError('transformations must be a list of prompt keys')
    cases = materialize_tests(original_image_path, transformations, output_dir,
                              num_tests=num_tests, seed=seed)
    return [MetamorphicTest(
        image_path=Path(case['image_paths'][0]), expected_answer='',
        description=case['description'],
        metadata={'transformation': case['description'], 'seed': seed},
    ) for case in cases]
