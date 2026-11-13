"""Contract for the future metamorphic generator; no model calls are made here."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union


@dataclass
class MetamorphicTest:
    """One materialized test, reused across all programs for a question.

    image_path points to the transformed image, not the original image.
    expected_answer must be supplied explicitly even for answer-changing tests.
    metadata can hold the transformation name, parameters, and generation details.
    """

    image_path: Path
    expected_answer: str
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
    """Generate up to num_tests materialized tests for one question.

    The future implementation should select and apply transformations to the
    original image, save transformed images under output_dir, and return their
    paths with expected answers. Returned paths should be absolute. The seed
    controls random choices; config contains generator-specific settings.

    Reuse original_answer only when a transformation preserves the answer.
    Answer-changing transformations require a derived or verified new answer.
    Never mutate the original image. The caller shares this suite across all
    candidate programs; generation does not receive a program or iteration.

    This interface does not yet adapt results to ViUniT's image groups or wire
    suite selection into generate.py/train.py.
    """
    raise NotImplementedError(
        "Implement per-question metamorphic test generation in "
        "unit_test_generation/metamorphic/generator.py."
    )
