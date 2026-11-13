"""Suite selection and cache management shared by evaluation and training."""
import hashlib
import json
from pathlib import Path

from .metamorphic import generator as metamorphic

CACHE_FIELDS = (
    'unit_tests', 'unit_test_prompt', 'selected_unit_tests', 'unit_test_images',
    'unit_test_results', 'unit_test_metadata', 'unit_test_suite_key',
)


def suite_settings(config):
    settings = dict(config.get('unit_test_suite', {}))
    mode = settings.get('mode', 'viunit')
    if mode not in {'viunit', 'metamorphic', 'hybrid'}:
        raise ValueError(f'Unknown unit_test_suite.mode: {mode}')
    if mode != 'viunit' and config['unit_test_generation']['use_program']:
        raise ValueError('Metamorphic and hybrid suites require use_program: false')
    default_count = config['unit_test_sampling']['num_unit_tests']
    settings.update(mode=mode)
    for name in ('viunit_count', 'metamorphic_count'):
        value = settings.get(name)
        settings[name] = default_count if value is None else value
        if not isinstance(settings[name], int) or isinstance(settings[name], bool) or settings[name] <= 0:
            raise ValueError(f'unit_test_suite.{name} must be a positive integer')
    return settings


def uses_viunit(config):
    return suite_settings(config)['mode'] in {'viunit', 'hybrid'}


def _plain(value):
    # Resolve OmegaConf interpolations without requiring it for interface tests.
    if type(value).__module__.startswith('omegaconf'):
        from omegaconf import OmegaConf
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _fingerprint(record, config, settings):
    image = Path(record['image_path']).resolve()
    stat = image.stat() if image.exists() else None
    sections = {name: _plain(config.get(name, {})) for name in (
        'unit_test_generation', 'unit_test_sampling', 'image_generation', 'execution',
    )}
    prompt_contents = {}
    for section in sections.values():
        for key, value in section.get('generation', {}).items():
            if key.endswith('_file') and value and Path(value).is_file():
                prompt_contents[str(value)] = Path(value).read_text()
    payload = dict(version=1, settings=_plain(settings), seed=config['seed'],
                   sections=sections, prompts=prompt_contents,
                   question_id=str(record.get('sample_id', record['index'])),
                   question=record.get('text', record.get('question')),
                   answer=record.get('answer'), image=str(image),
                   image_stat=(stat.st_size, stat.st_mtime_ns) if stat else None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def invalidate_suite_cache(dataset, config):
    """Drop incompatible suites before cached feedback can influence programs."""
    settings = suite_settings(config)
    if config['unit_test_generation']['use_program']:
        return  # Preserve the existing program-conditioned ViUniT path.
    keys = [_fingerprint(dataset[i], config, settings) for i in range(len(dataset))]
    if any(d.get('unit_test_suite_key') != key or
           'selected_unit_tests' not in d or 'unit_test_images' not in d
           for d, key in zip(dataset.data, keys)):
        for d in dataset.data:
            for field in CACHE_FIELDS:
                d.pop(field, None)


def build_suite(dataset, config, viunit_provider):
    """Populate executor-compatible fields once per question.

    viunit_provider runs the existing batched ViUniT pipeline without changes.
    Unversioned caches are rebuilt once because their settings cannot be verified.
    A batch is rebuilt together to support the legacy provider's batch guards.
    """
    settings = suite_settings(config)
    records = [dataset[i] for i in range(len(dataset))]
    keys = [_fingerprint(record, config, settings) for record in records]
    if all(d.get('unit_test_suite_key') == key and
           'selected_unit_tests' in d and 'unit_test_images' in d
           for d, key in zip(dataset.data, keys)):
        return

    for d in dataset.data:
        for field in CACHE_FIELDS:
            d.pop(field, None)

    # Build custom tests first so an unimplemented generator fails before diffusion.
    custom = []
    if settings['mode'] in {'metamorphic', 'hybrid'}:
        for record, key in zip(records, keys):
            output = Path(config['output_dir']).resolve() / 'metamorphic' / key
            output.mkdir(parents=True, exist_ok=True)
            tests = metamorphic.generate_tests(
                question_id=str(record.get('sample_id', record['index'])),
                question=record.get('text', record.get('question')),
                original_image_path=Path(record['image_path']).resolve(),
                original_answer=record['answer'],
                num_tests=settings['metamorphic_count'], output_dir=output,
                seed=config['seed'], config=settings.get('metamorphic', {}),
            )
            if not tests or len(tests) > settings['metamorphic_count']:
                raise ValueError('Metamorphic generator must return 1..metamorphic_count tests')
            for test in tests:
                if not Path(test.image_path).is_absolute() or not Path(test.image_path).is_file():
                    raise ValueError(f'Metamorphic image must be an existing absolute path: {test.image_path}')
                if not isinstance(test.expected_answer, str):
                    raise ValueError('Metamorphic expected_answer must be a string')
            custom.append(tests)

    if uses_viunit(config):
        original_count = config['unit_test_sampling']['num_unit_tests']
        try:
            config['unit_test_sampling']['num_unit_tests'] = settings['viunit_count']
            viunit_provider()
        finally:
            config['unit_test_sampling']['num_unit_tests'] = original_count

    for i, (d, key) in enumerate(zip(dataset.data, keys)):
        d.setdefault('unit_tests', [])
        selected = d.setdefault('selected_unit_tests', [])
        images = d.setdefault('unit_test_images', [])
        metadata = [{'source': 'viunit'} for _ in selected]
        if custom:
            for test in custom[i]:
                selected.append((test.description, test.expected_answer))
                images.append([str(test.image_path)])
                metadata.append({**test.metadata, 'source': 'metamorphic'})
        if not selected or len(selected) != len(images):
            raise ValueError('Suite must contain tests with aligned image groups')
        d['unit_test_metadata'] = metadata
        d['unit_test_suite_key'] = key
