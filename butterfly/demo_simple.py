#!/usr/bin/env python3
"""Minimal ViUniT text unit-test generation smoke test (no diffusion/Gradio).

ViUniT generates textual (question, expected_answer) tests from a QUESTION,
not directly from the image. The --image argument checks the source image is
readable, but its pixels are NOT passed to the text-generation model.
"""
import argparse
import copy
import json
import os
from pathlib import Path

PROJECT = Path('/work/11808/shumera/ls6/projects/butterfly')
DEFAULT_IMAGE = Path('/work/11808/shumera/ls6/butterfly_shared_resources/datasets/coco/val2017/000000482319.jpg')
DEFAULT_CONFIG = PROJECT / 'viunit_configs/base.yaml'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=DEFAULT_IMAGE)
    parser.add_argument('--question', default='How many people are in the image?',
                        help='Question whose ViUniT text unit tests should be generated')
    parser.add_argument('--output', type=Path, default=PROJECT / 'plots/simple_unit_tests.json')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--num-tests', type=int, default=2)
    args = parser.parse_args()

    # Validate the input image before loading the larger model dependencies.
    from PIL import Image
    with Image.open(args.image) as img:
        img.verify()
    print(f'Input image readable: {args.image}', flush=True)
    print('Note: ViUniT text test generation uses --question, not the image pixels.', flush=True)

    os.environ['CONFIG_NAMES'] = 'demo_config'
    if os.environ.get('CONDA_PREFIX'):
        os.environ.setdefault('CUDA_HOME', os.environ['CONDA_PREFIX'])

    import omegaconf
    from transformers import AutoTokenizer
    from unit_test_generation.processing import get_unit_test_prompt, extract_unit_tests
    from unit_test_generation.unit_test_sampling import TextSampler
    from viper_configs import viper_config
    from main_simple_lib import forward

    cfg = omegaconf.OmegaConf.load(args.config)
    generation = cfg['unit_test_generation']['generation']
    with open(generation['prompt_file'], encoding='utf-8') as f:
        system_prompt = f.read()
    with open(generation['in_context_examples_file'], encoding='utf-8') as f:
        examples = f.read()

    model_name = viper_config.llm.model_id
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=os.getenv('HF_ACCESS_TOKEN'))
    sampler = TextSampler(
        model_name=cfg['unit_test_sampling']['model_name'],
        sampling_strategy=cfg['unit_test_sampling']['strategy'],
        filter_long_answers=cfg['unit_test_sampling']['filter_long_answers'])
    prompts = get_unit_test_prompt(
        [args.question], system_prompt, examples, model_name, tokenizer)

    tests = []
    for attempt in range(2):
        print(f'Generating unit tests (attempt {attempt + 1})...', flush=True)
        response = forward(
            model_name='llm_general', prompt=copy.deepcopy(prompts), queues=None,
            min_new_tokens=10, max_new_tokens=180, return_full_text=False,
            top_p=0.9, do_sample=True, num_return_sequences=3)
        if not isinstance(response, str):
            raise TypeError(f'Unexpected forward() output type: {type(response).__name__}')
        tests = extract_unit_tests(response.split('assistant')[-1])
        if tests:
            break
    if not tests:
        raise RuntimeError('No parseable unit tests returned from the LLM')

    tests = sampler.sample(tests, args.num_tests)
    records = [{'question': str(test[0]), 'expected_answer': str(test[1])}
               for test in tests]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'source_image': str(args.image),
        'original_question': args.question,
        'note': 'Source image is not used to condition ViUniT text unit-test generation.',
        'unit_tests': records,
    }, indent=2) + '\n', encoding='utf-8')
    for i, record in enumerate(records, 1):
        print(f"Test {i}: {record['question']} -> {record['expected_answer']}", flush=True)
    print(f'Saved to {args.output}', flush=True)


if __name__ == '__main__':
    main()

