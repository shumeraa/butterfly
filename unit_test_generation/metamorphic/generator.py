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
    expected_answer: Union[str, Dict[str, str]]
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

    predictions_path reads the merged LLM format (question_id and parsed).
    tests_path reads the custom-test JSON mapping produced by apply_transformations.
    Otherwise transformations selects prompt keys, defaulting to all six.
    Blank answers and mapping rules are retained for candidate instrumentation.
    """
    from apply_transformations.apply import eligible_transformations, materialize_tests
    from image_transforms import HIGH_LEVEL_TRANSFORMS

    predictions_path = config.get('predictions_path')
    tests_path = config.get('tests_path')
    if predictions_path and tests_path:
        raise ValueError('Choose predictions_path or tests_path, not both')
    if tests_path:
        cases = json.loads(Path(tests_path).read_text(encoding='utf-8'))[str(question_id)]
        tests = []
        for case in cases[:num_tests]:
            paths = case['image_paths']
            if not isinstance(paths, list) or len(paths) != 1:
                raise ValueError('Metamorphic tests require exactly one transformed image per test')
            tests.append(MetamorphicTest(
                image_path=Path(paths[0]).resolve(), expected_answer=case.get('answer', ''),
                description=case['description'], metadata={'input': 'tests_path'},
            ))
        return tests

    transformations = config.get('transformations', HIGH_LEVEL_TRANSFORMS)
    if predictions_path:
        predictions = json.loads(Path(predictions_path).read_text(encoding='utf-8'))
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
