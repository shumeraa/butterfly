"""Convert LLM effect labels into custom-test rules and choose a small suite."""
import copy
import random
from collections.abc import Mapping

from unit_test_generation.metamorphic.baselines import validate_answer_rule


def tests_from_labels(labels):
    """Flatten MR1/MR2 labels without inventing tests or changing answer mappings."""
    if not isinstance(labels, list):
        raise ValueError('Labels must be a list of [question_id, effects] pairs')
    result = {}
    for row in labels:
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[1], Mapping):
            raise ValueError('Each label must be [question_id, effects]')
        question_id, effects = str(row[0]), row[1]
        if question_id in result:
            raise ValueError(f'Duplicate question ID: {question_id}')
        tests = result[question_id] = []

        def add(description, operation, effect, **parameters):
            if effect is None or effect == 'unsafe' or effect == {}:
                return
            if effect == 'invariant':
                answer, relation = '', 'invariant'
            elif isinstance(effect, Mapping) and 'equivariant' in effect:
                answer, relation = effect['equivariant'], 'equivariant'
                if not isinstance(answer, Mapping) or not answer:
                    raise ValueError(f'Expected a nonempty answer mapping for {question_id}: {description}')
            else:
                raise ValueError(f'Unknown effect for {question_id}: {description}: {effect!r}')
            validate_answer_rule(answer)
            tests.append(dict(description=description, answer=copy.deepcopy(answer),
                              transformation=operation, relation=relation, parameters=parameters))

        for name, effect in effects.items():
            if name == 'mr2':
                if effect is None or effect == 'unsafe' or effect == {}:
                    continue
                if not isinstance(effect, Mapping):
                    raise ValueError(f'mr2 must be an object or null for {question_id}')
                for item, operations in effect.items():
                    if operations is None:
                        continue
                    for color, color_effect in (operations.get('fill_color') or {}).items():
                        add(f'{item} turned {color}', 'fill_color', color_effect, item=item, color=color)
                    add(f'{item}, occluded bounding box', 'occlude_bbox',
                        operations.get('occlude_bbox'), item=item)
            elif name in {'hflip', 'vflip', 'grayscale'}:
                add(name, name, effect)
            else:
                raise ValueError(f'Unsupported label operation: {name}')
    return result


def select_label_tests(tests, *, question_id, max_tests=3, seed=42):
    """Prioritize equivariance and fill unused slots with seeded invariant samples."""
    if not isinstance(max_tests, int) or isinstance(max_tests, bool) or max_tests < 1:
        raise ValueError('max_tests must be a positive integer')
    randomizer = random.Random(f'{seed}:{question_id}')
    equivariant = [test for test in tests if test['relation'] == 'equivariant']
    invariant = [test for test in tests if test['relation'] == 'invariant']
    if len(equivariant) > max_tests:
        return randomizer.sample(equivariant, max_tests)
    remaining = min(max_tests - len(equivariant), len(invariant))
    return equivariant + randomizer.sample(invariant, remaining)
