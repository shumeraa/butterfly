"""Candidate-specific expectations without mutating shared metamorphic rules."""
import copy
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path


def answer_options(expected):
    """A list represents alternatives, rather than one literal list-valued answer."""
    return expected if isinstance(expected, list) else [expected]


def validate_answer_rule(rule):
    if isinstance(rule, str):
        return
    if isinstance(rule, Mapping) and all(
        isinstance(key, str) and (
            isinstance(value, str) or (isinstance(value, list) and value
                                      and all(isinstance(item, str) for item in value))
        ) for key, value in rule.items()
    ):
        return
    raise ValueError('Custom answer must be a string or a mapping to strings/nonempty string lists')


class _Skip:
    """Sentinel: this candidate cannot be checked by this test, so it is neither passed nor failed."""
    def __repr__(self):
        return 'SKIP'


SKIP = _Skip()
UNCHECKED = 'unchecked'   # judge notation for "skip this input"; never an expected answer
SKIP_PREFIX = 'SKIP: '    # marks a skipped test in the errors dict returned by instrument_tests


def _norm(text):
    return re.sub(r'[^a-z ]', '', str(text).lower()).strip()


def _lookup_key(rule, output):
    """Exact key first, then a normalized match, then a short answer containing exactly one key."""
    if output in rule:
        return output
    normalized = {_norm(key): key for key in rule if key != 'any'}
    if _norm(output) in normalized:
        return normalized[_norm(output)]
    words = _norm(output).split()
    if 0 < len(words) <= 3:
        found = {key for norm_key, key in normalized.items() if norm_key and norm_key in words}
        if len(found) == 1:
            return found.pop()
    return None


def resolve_answer(rule, original_output):
    """Expected answer for one candidate, or SKIP when the rule gives it nothing to check.

    A mapping without a matching key (or `any`) is SKIP rather than "unchanged": the rule says nothing
    about that answer. An `unchecked` value is SKIP as well.
    """
    validate_answer_rule(rule)
    if isinstance(rule, Mapping):
        if not rule:
            return original_output   # an empty mapping preserves the baseline, like an empty string
        key = _lookup_key(rule, original_output)
        value = rule[key] if key is not None else rule.get('any', SKIP)
        if value is SKIP:
            return SKIP
        return SKIP if UNCHECKED in answer_options(value) else copy.deepcopy(value)
    if rule == UNCHECKED:
        return SKIP
    return original_output if rule == '' else rule


def score_expected(output, expected, accuracy, checker=None):
    """Accept any mapped alternative, using the existing scalar scorer for each."""
    return max((checker.are_synonymous(output, option) if checker else accuracy([output], [option]))
               for option in answer_options(expected))


def execution_body(code, extract_code):
    body = extract_code(code)
    for signature in ('def execute_command(image) -> str:\n',
                      'def execute_command(image):\n'):
        if signature in body:
            body = body.split(signature, 1)[1]
            break
    return body


def collect_baselines(dataset, iteration, config, execute, extract_code):
    """Execute uncached candidates on original images and return a shared dict.

    Persist results on dataset records so data.p can resume them. Cache identity
    includes program, question, source image stat, execution settings, and seed.
    """
    shared, requests, pending = {}, [], []
    for index in range(len(dataset)):
        record = dataset.data[index]
        question_id = str(record.get('sample_id', record.get('questionId', record['index'])))
        image = Path(dataset[index]['image_path']).resolve()
        stat = image.stat() if image.exists() else None
        cache = record.setdefault('original_program_outputs', {}).setdefault(iteration, {})
        for candidate, code in enumerate(record.get('generated_code', {}).get(iteration, [])):
            body = execution_body(code, extract_code)
            payload = dict(code=body, image=str(image), question=record['question'],
                           image_stat=(stat.st_size, stat.st_mtime_ns) if stat else None,
                           seed=config['seed'], execution=dict(config.get('execution', {})))
            fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                                   default=str).encode()).hexdigest()
            key = (question_id, iteration, candidate)
            previous = cache.get(candidate)
            if previous and previous.get('fingerprint') == fingerprint:
                shared[key] = previous
                continue
            # A changed baseline makes previously scored tests incompatible.
            record.get('unit_test_results', {}).pop(iteration, None)
            record.get('candidate_unit_tests', {}).pop(iteration, None)
            sample_id = f'baseline_{index}_{iteration}_{candidate}'
            requests.append(dict(sample_id=sample_id, answer='', possible_answers=[''],
                                 query_type='image', codes=body,
                                 query=record['question'], image=str(image)))
            pending.append((cache, candidate, fingerprint, key, sample_id))
    try:
        results = execute(requests, config, use_fixed_code=None) if requests else {}
    except Exception as error:
        # Preserve baseline failures as test failures rather than guessing answers.
        results = {request['sample_id']: dict(output=None,
                   error=f'{type(error).__name__}: {error}') for request in requests}
    for cache, candidate, fingerprint, key, sample_id in pending:
        result = copy.deepcopy(results.get(sample_id, {
            'output': None, 'error': 'Baseline executor returned no result',
        }))
        if result.get('error') is None and not isinstance(result.get('output'), str):
            result['error'] = 'Baseline executor returned no string answer'
        result['fingerprint'] = fingerprint
        cache[candidate] = result
        shared[key] = result
    return shared


def instrument_tests(record, tests, iteration, candidate, enabled, original_answers):
    """Return candidate expectations and per-test baseline errors; keep rules shared."""
    resolved, errors = [], {}
    metadata = record.get('unit_test_metadata', [])
    question_id = str(record.get('sample_id', record.get('questionId', record['index'])))
    baseline = original_answers.get((question_id, iteration, candidate))
    for index, (description, rule) in enumerate(tests):
        custom = index < len(metadata) and metadata[index].get('source') == 'metamorphic'
        expected = rule
        if custom and enabled:
            if baseline is None or baseline.get('error') is not None:
                error = baseline.get('error') if baseline else 'Missing original execution result'
                errors[index] = f'Baseline execution failed: {error}'
                expected = ''
            else:
                expected = resolve_answer(rule, baseline['output'])
                if expected is SKIP:
                    errors[index] = SKIP_PREFIX + 'rule does not constrain this candidate\'s baseline answer'
                    expected = ''
        elif isinstance(rule, Mapping):
            raise ValueError('Answer mappings require execution.metamorphic_baselines: true')
        resolved.append((description, expected))
    if enabled:
        record.setdefault('candidate_unit_tests', {}).setdefault(iteration, {})[candidate] = resolved
    return resolved, errors


def is_skipped(message):
    return str(message).startswith(SKIP_PREFIX)


def skipped_test(message):
    """Placeholder that keeps test indices aligned but is excluded from the program's mean."""
    return dict(acc=0., skipped=True, error=None, results=[], reason=message)


def failed_baseline_test(message):
    """A failed baseline is a failed test even when its image group is empty."""
    return dict(acc=0., error=message, failed=True, baseline_error=True, results=[
        dict(output=None, error=message, acc=0., expected_answer=None, baseline_error=True)
    ])


def baseline_error_result(message):
    """Result entry for a test that was not executed: skipped by the rule, or a failed baseline."""
    return skipped_test(message) if is_skipped(message) else failed_baseline_test(message)


def mean_scored(results):
    """Program score: mean over tests that were actually scored; skipped tests are excluded.

    A program with nothing scored gets 0.0, as a program with no tests did before.
    """
    scores = [r['acc'] for r in results if not r.get('skipped')]
    return sum(scores) / len(scores) if scores else 0.0
